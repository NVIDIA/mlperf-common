#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage a dataset from shared storage onto node-local storage, collectively.

Every node needs the same bytes.  Having each node read the whole dataset from
Lustre is N-times redundant and runs at the speed of the slowest reader, so
instead each rank reads a disjoint shard and the bytes cross the fabric once,
via NCCL.

Decomposition (W = world size, L = ranks per node, N = nodes):

  * The world is split into L process groups, group `l` holding the rank with
    LOCAL_RANK == l on every node.  Group `l` owns slice `l` of the file -- a
    contiguous 1/L of it.
  * Inside group `l`, each of the N ranks reads a disjoint 1/N sub-shard of
    that slice from shared storage, and an all-gather assembles the whole slice
    on every node.  Each rank then writes slice `l` to its node-local copy.
  * The node's L ranks together write the whole file, so nothing is exchanged
    or written twice within a node.

That gives W disjoint readers on the shared filesystem, one fabric crossing per
byte per node, and L concurrent all-gathers which between them drive every NIC
without any hand-tuned per-cluster transport configuration.

Streaming is windowed so memory stays bounded, and reads, the collective, and
writes are pipelined across rounds.

Usage mirrors cp/rsync and `fastcp`:

    python3 -m mlperf_common.fileio.datastage -r SRC... DST

Launch with one task per GPU, under slurm2pytorch so that RANK / WORLD_SIZE /
LOCAL_RANK / MASTER_ADDR are set:

    srun --ntasks-per-node=${DGXNGPU} ... slurm2pytorch \\
        python3 -m mlperf_common.fileio.datastage -r "${SLOW_DATADIR}/${DATASET}" "${DATADIR}"
"""

import argparse
import os
import queue
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

from mlperf_common.fileio import direct_io
from mlperf_common.fileio.copyplan import (
    CopyArgumentError, UnreadableEntries, plan_copy_operations, validate_copy_args)

# Buffer alignment.  2 MiB is the Linux huge page size, comfortably above any
# filesystem block size we will meet, so O_DIRECT is always happy.  Same
# constant fastcp uses.
BUFFER_ALIGN = 2 * 1024 * 1024

# Don't bother splitting a read across threads below this; the syscall is
# already large enough to saturate a reader.
MIN_READ_PIECE = 4 * 1024 * 1024

# Pipeline depth.  Send slots let the reader run ahead of the collective.
SEND_SLOTS = 3

# Assembled windows held on the device.  Two, so the next round's all-gather
# can run while the previous window is still being copied back.  Device memory
# is the cheap resource here: this runs as its own job step, before training,
# and everything is released when the process exits.
RECV_DEV_SLOTS = 2

# Host staging for the write side.  An assembled window is a concatenation of
# one segment per node, each bound for a different file offset, so it can be
# copied back and written a few segments at a time instead of whole.  Keeping
# this pool small and fixed is what stops pinned host memory -- which is
# page-locked and reserved up front -- from scaling with the node count.
CHUNK_TARGET = 64 << 20
CHUNK_SLOTS = 3


def ceil_div(a, b):
    return -(-a // b)


def align_up(value, multiple):
    return ceil_div(value, multiple) * multiple


def align_down(value, multiple):
    return (value // multiple) * multiple


def parse_size(text):
    """Parse a size with an optional K/M/G/T suffix."""
    text = str(text).strip()
    multipliers = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}
    multiplier = 1
    if text and text[-1].lower() in multipliers:
        multiplier = multipliers[text[-1].lower()]
        text = text[:-1]
    value = int(float(text) * multiplier)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"size must be positive: {text}")
    return value


def open_maybe_direct(path, flags):
    """Open with O_DIRECT, falling back to buffered if the filesystem says no.

    Reads and writes stay block-padded either way, which is harmless for a
    buffered descriptor, so nothing downstream has to care which we got.
    """
    try:
        return os.open(path, flags | os.O_DIRECT)
    except OSError as exc:
        if exc.errno not in (22, 95):  # EINVAL, EOPNOTSUPP
            raise
        return os.open(path, flags)


def pinned_aligned(nbytes, alignment):
    """Allocate a pinned host buffer aligned for O_DIRECT.

    cudaHostAlloc returns page-aligned memory in practice, but O_DIRECT is
    unforgiving and the failure mode (EINVAL deep in a worker thread) is
    miserable to debug, so over-allocate and slice to a known-aligned offset
    rather than trusting it.

    Returns (owner, view, memoryview); `owner` must be kept alive.
    """
    owner = torch.empty(nbytes + alignment, dtype=torch.uint8, pin_memory=True)
    offset = (-owner.data_ptr()) % alignment
    view = owner[offset:offset + nbytes]
    assert view.data_ptr() % alignment == 0, "failed to align pinned buffer"
    return owner, view, memoryview(view.numpy())


class Topology:
    """Rank/node layout and the per-LOCAL_RANK process groups."""

    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

        # Derive the node layout from what the ranks actually report rather
        # than assuming a block distribution of RANK: run.sub launches some
        # steps with --distribution=arbitrary, and silently mis-grouping would
        # produce a corrupt copy rather than an error.
        identities = [None] * self.world_size
        dist.all_gather_object(identities, (socket.gethostname(), self.local_rank))

        hosts = []
        for host, _ in identities:
            if host not in hosts:
                hosts.append(host)
        self.node_count = len(hosts)
        node_index = {host: i for i, host in enumerate(hosts)}

        # group_ranks[l][node] = global rank of LOCAL_RANK l on that node
        group_ranks = [[None] * self.node_count for _ in range(self.local_world_size)]
        for global_rank, (host, local_rank) in enumerate(identities):
            if local_rank >= self.local_world_size:
                raise RuntimeError(
                    f"rank {global_rank} reports LOCAL_RANK={local_rank} with "
                    f"LOCAL_WORLD_SIZE={self.local_world_size}"
                )
            slot = group_ranks[local_rank][node_index[host]]
            if slot is not None:
                raise RuntimeError(
                    f"ranks {slot} and {global_rank} both claim LOCAL_RANK="
                    f"{local_rank} on {host}"
                )
            group_ranks[local_rank][node_index[host]] = global_rank
        for local_rank, ranks in enumerate(group_ranks):
            if any(r is None for r in ranks):
                raise RuntimeError(
                    f"not every node has a rank with LOCAL_RANK={local_rank}; "
                    "launch with a uniform --ntasks-per-node"
                )

        # new_group is collective: every rank creates every group, in the same
        # order, but only ever uses its own.
        self.groups = [dist.new_group(ranks) for ranks in group_ranks]
        self.group = self.groups[self.local_rank]
        self.node_index = node_index[socket.gethostname()]

    def describe(self):
        return (
            f"{self.world_size} ranks = {self.node_count} nodes x "
            f"{self.local_world_size} ranks/node"
        )


class FileLayout:
    """Deterministic, identical-on-every-rank mapping of a file onto ranks.

    Slice `l` of the file belongs to LOCAL_RANK l's group; within that group,
    node `i` reads sub-shard `i` and the all-gather hands every node the whole
    slice, one `window` at a time.

    Every offset and length here is a multiple of `align` except the very last
    byte range of the file.  That matters: O_DIRECT writes are padded up to the
    block size, so a padded write must never reach into a range some other rank
    owns.  Because only the final range of the final slice is unaligned, the
    only padding that ever happens runs off the end of the file, and the
    closing ftruncate trims it.
    """

    def __init__(self, size, topology, piece, align):
        self.size = size
        self.align = align
        self.piece = piece
        L = topology.local_world_size
        N = topology.node_count

        self.slice_size = align_up(ceil_div(size, L), align) if size else 0
        self.slice_start = min(topology.local_rank * self.slice_size, size)
        slice_end = min(self.slice_start + self.slice_size, size)
        self.slice_len = max(slice_end - self.slice_start, 0)

        # Per-node sub-shard of this slice.
        self.shard = align_up(ceil_div(self.slice_len, N), align) if self.slice_len else 0
        self.rounds = ceil_div(self.shard, self.piece) if self.shard else 0

    def segment(self, node, round_index):
        """(file_offset, length) that `node` contributes in `round_index`."""
        shard_start = min(node * self.shard, self.slice_len)
        shard_len = min(self.shard, self.slice_len - shard_start)
        base = round_index * self.piece
        if base >= shard_len:
            return self.slice_start + shard_start + base, 0
        length = min(self.piece, shard_len - base)
        return self.slice_start + shard_start + base, length


class Stager:
    def __init__(self, args, topology):
        self.args = args
        self.topo = topology
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.pool = ThreadPoolExecutor(max_workers=max(args.num_threads, 4))

        self.align = BUFFER_ALIGN
        self.dest_root = os.path.abspath(args.destination)
        # buffer_size is what each rank reads per round, exactly as fastcp's
        # --buffer-size is what each thread reads.  The assembled window that
        # every rank must hold is node_count of those, so memory scales with the
        # job: an all-gather delivers the whole window to every participant.
        self.piece = args.buffer_size
        window = self.piece * topology.node_count
        # How many segments come back from the device per host staging chunk.
        self.group_nodes = max(1, min(topology.node_count, CHUNK_TARGET // self.piece))
        chunk_bytes = self.group_nodes * self.piece

        # Fail before allocating rather than dying inside a CUDA OOM.  Only the
        # device side scales with the node count now, so this is the check that
        # matters; 60% leaves room for the NCCL buffers and the CUDA context.
        device_bytes = self.piece + RECV_DEV_SLOTS * window
        budget = int(torch.cuda.get_device_properties(self.device).total_memory * 0.60)
        if device_bytes > budget:
            per_window = max(align_down(budget // (RECV_DEV_SLOTS * topology.node_count),
                                        self.align), self.align)
            raise RuntimeError(
                f"--buffer-size {self.piece >> 20}M across {topology.node_count} nodes "
                f"needs {device_bytes / 1024 ** 3:.1f} GiB of device memory, over the "
                f"{budget / 1024 ** 3:.1f} GiB budget. Lower --buffer-size to at most "
                f"{per_window >> 20}M."
            )

        self.send_host = [pinned_aligned(self.piece, self.align) for _ in range(SEND_SLOTS)]
        self.chunk_host = [pinned_aligned(chunk_bytes, self.align) for _ in range(CHUNK_SLOTS)]
        self.send_dev = torch.empty(self.piece, dtype=torch.uint8, device=self.device)
        self.recv_dev = [torch.empty(window, dtype=torch.uint8, device=self.device)
                         for _ in range(RECV_DEV_SLOTS)]

        # The copy-back gets a stream of its own.  non_blocking=True only means
        # the host does not wait; the copy still queues on whatever stream it
        # was issued to, so on the default stream it runs after the next round's
        # collective rather than beside it on the copy engine -- which leaves
        # the second window RECV_DEV_SLOTS pays for doing nothing.
        self.drain_stream = torch.cuda.Stream(device=self.device)

        if topology.rank == 0:
            pinned = SEND_SLOTS * self.piece + CHUNK_SLOTS * chunk_bytes
            print(f"datastage: {self.piece >> 20} MiB per rank per round, "
                  f"{window / 1024 ** 3:.2f} GiB window; per rank "
                  f"{device_bytes / 1024 ** 3:.2f} GiB device, "
                  f"{pinned / 1024 ** 3:.2f} GiB pinned host "
                  f"({self.group_nodes} segments per {chunk_bytes >> 20} MiB chunk)",
                  flush=True)

    def _read_piece(self, fd, mview, offset, length, block_size):
        """Fill mview[:length] from fd at offset, split across reader threads."""
        if length == 0:
            return
        nthreads = max(1, min(self.args.num_threads, length // MIN_READ_PIECE))
        if nthreads == 1:
            direct_io.pread(fd, mview, length, offset, block_size, 0)
            return
        # Aligning the per-thread piece to `align` (not to some larger read
        # block) is what keeps this from silently collapsing to one thread when
        # the round is small.
        per = max(align_down(length // nthreads, self.align), self.align)
        futures = []
        start = 0
        tid = 0
        while start < length:
            count = min(per, length - start)
            # Hand each read the whole remaining buffer, not just `count`:
            # direct_io pads the request up to the block size and asserts the
            # view is big enough to hold the padding.
            futures.append(
                self.pool.submit(
                    direct_io.pread, fd, mview[start:], count,
                    offset + start, block_size, tid,
                )
            )
            start += count
            tid += 1
        for future in futures:
            future.result()

    def _write_one(self, ready, fd, mview, base, stride, length, offset, block_size):
        """Write one segment, once the copy that filled its chunk has landed.

        Waiting here rather than on the draining thread means the next chunk's
        device-to-host copy can be issued while this one is still being written.
        """
        ready.synchronize()
        return direct_io.pwrite(fd, mview[base:base + stride], length, offset, block_size)

    def stage_file(self, src, dst, size, mtime_ns):
        topo = self.topo
        tmp = f"{dst}.datastage.tmp.{os.environ.get('SLURM_JOB_ID', 'nojob')}"

        # One rank per node creates the file, so the others can open it without
        # racing on O_CREAT and without truncating each other's writes.
        if topo.local_rank == 0:
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            self._chmod_parents(dst)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, self.args.chmod)
            try:
                if size:
                    os.ftruncate(fd, size)
                os.fchmod(fd, self.args.chmod)
            finally:
                os.close(fd)
        dist.barrier()

        fd_src = open_maybe_direct(src, os.O_RDONLY)
        fd_dst = open_maybe_direct(tmp, os.O_WRONLY)
        try:
            src_block = os.fstatvfs(fd_src).f_bsize
            dst_block = os.fstatvfs(fd_dst).f_bsize
            assert self.align >= max(src_block, dst_block), "alignment below fs block size"

            layout = FileLayout(size, topo, self.piece, self.align)
            if layout.rounds:
                self._run_pipeline(fd_src, fd_dst, layout, src_block, dst_block)
            os.fsync(fd_dst)
        finally:
            os.close(fd_src)
            os.close(fd_dst)

        # Publish only once every rank on every node has written and synced.
        dist.barrier()
        if topo.local_rank == 0:
            # O_DIRECT writes are padded up to the block size, so the last write
            # of the file runs past its end.  Trim it back -- after the barrier,
            # because until then another rank may still be padding.
            fd = os.open(tmp, os.O_WRONLY)
            try:
                os.ftruncate(fd, size)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.utime(tmp, ns=(mtime_ns, mtime_ns))
            os.rename(tmp, dst)
            parent = os.open(os.path.dirname(dst), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)

    def _chmod_parents(self, dst):
        """Make the directories we created world-writable.

        os.makedirs applies the umask, which would leave subdirectories 0755 on
        node-local scratch that is shared between users -- the same trap the
        rsync path avoided with `umask 0000 --chmod=ugo+rwx`.

        Bounded to the destination root: these steps run with
        --container-remap-root, so walking further up would happily chmod real
        directories outside the staging tree.
        """
        path = os.path.dirname(dst)
        while path.startswith(self.dest_root + os.sep) or path == self.dest_root:
            try:
                os.chmod(path, self.args.chmod)
            except OSError:
                break
            path = os.path.dirname(path)

    def _run_pipeline(self, fd_src, fd_dst, layout, src_block, dst_block):
        topo = self.topo
        free_q = queue.Queue()
        filled_q = queue.Queue()
        recv_free_q = queue.Queue()
        drain_q = queue.Queue()
        for slot in range(SEND_SLOTS):
            free_q.put(slot)
        for slot in range(RECV_DEV_SLOTS):
            recv_free_q.put(slot)
        failure = []
        stop = threading.Event()

        def reader():
            try:
                for round_index in range(layout.rounds):
                    slot = free_q.get()
                    if stop.is_set():
                        break
                    offset, length = layout.segment(topo.node_index, round_index)
                    _, _, mview = self.send_host[slot]
                    self._read_piece(fd_src, mview, offset, length, src_block)
                    filled_q.put((round_index, slot, length))
            except BaseException as exc:  # noqa: BLE001 - re-raised on main thread
                failure.append(exc)
            finally:
                filled_q.put(None)

        def drainer():
            """Copy assembled windows back a chunk at a time and write them.

            Runs off the main thread so that the collective for round k+1 can
            overlap the copy-back and writes for round k.  A device window is
            only released once all of its copies have landed; the writes carry
            on behind that against the host chunk pool.
            """
            # CUDA's current device is per host thread, and main() set it on
            # the main thread only, so this one starts out on device 0.  The
            # copies below would still run on the right device -- the tensors
            # carry it -- but torch.cuda.Event binds to the *calling thread's*
            # current device when recorded, so the events would land on device
            # 0's idle stream and every synchronize() on them would return
            # immediately, letting the writers read chunks the copies have not
            # filled yet.
            torch.cuda.set_device(self.device)

            chunk_pending = [[] for _ in range(CHUNK_SLOTS)]
            try:  # noqa: PLR1702
                while True:
                    item = drain_q.get()
                    if item is None:
                        break
                    dev_slot, round_index, assembled = item
                    # Order the copies behind the collective on the device
                    # rather than on the host: the copies below can be queued
                    # while the all-gather that fills this window is still
                    # running, and the GPU holds them until it finishes.
                    self.drain_stream.wait_event(assembled)
                    last_copy = None
                    for group, first in enumerate(
                            range(0, topo.node_count, self.group_nodes)):
                        # Deliberately not `stop`: that only means the main loop
                        # has stopped feeding us, which is the normal end of a
                        # file. Windows already queued still have to be written.
                        if failure:
                            break
                        count = min(self.group_nodes, topo.node_count - first)
                        nbytes = count * layout.piece
                        chunk = group % CHUNK_SLOTS
                        for future in chunk_pending[chunk]:
                            future.result()
                        chunk_pending[chunk] = []

                        _, chunk_view, chunk_mview = self.chunk_host[chunk]
                        base = first * layout.piece
                        with torch.cuda.stream(self.drain_stream):
                            chunk_view[:nbytes].copy_(
                                self.recv_dev[dev_slot][base:base + nbytes],
                                non_blocking=True)
                            copied = torch.cuda.Event(blocking=True)
                            copied.record()
                        last_copy = copied

                        submitted = []
                        for index in range(count):
                            offset, length = layout.segment(first + index, round_index)
                            if length == 0:
                                continue
                            submitted.append(self.pool.submit(
                                self._write_one, copied, fd_dst, chunk_mview,
                                index * layout.piece, layout.piece, length, offset,
                                dst_block))
                        chunk_pending[chunk] = submitted

                    # The device window is reusable once its copies have landed,
                    # which is well before the writes behind them finish.
                    #
                    # This is also what keeps the window safe now that the
                    # copies read it from their own stream: the next all-gather
                    # is issued on the default stream, which has no ordering
                    # against drain_stream, so the only thing stopping it from
                    # overwriting a window still being copied is that the slot
                    # does not go back on recv_free_q until the host has seen
                    # the last copy complete.
                    if last_copy is not None:
                        last_copy.synchronize()
                    recv_free_q.put(dev_slot)
                for futures in chunk_pending:
                    for future in futures:
                        future.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised on main thread
                failure.append(exc)
                stop.set()
                # Never leave the main thread parked waiting for a window.
                for _ in range(RECV_DEV_SLOTS):
                    recv_free_q.put(0)
            finally:
                # No write may still be in flight when we return: stage_file
                # closes the destination fd as soon as _run_pipeline does.
                for futures in chunk_pending:
                    for future in futures:
                        try:
                            future.result()
                        except BaseException:  # noqa: BLE001 - already recorded
                            pass

        reader_thread = threading.Thread(target=reader, name="datastage-reader")
        drain_thread = threading.Thread(target=drainer, name="datastage-drainer")
        reader_thread.start()
        drain_thread.start()

        try:
            for round_index in range(layout.rounds):
                item = filled_q.get()
                if item is None or failure:
                    break
                _, slot, length = item

                _, send_view, _ = self.send_host[slot]
                self.send_dev.copy_(send_view, non_blocking=True)
                h2d_done = torch.cuda.Event(blocking=True)
                h2d_done.record()

                # Safe to launch before the copy has landed: the collective is
                # queued behind it on the same stream.  Waiting on this one
                # event, rather than the whole device, is what lets the reader
                # refill the slot while the collective is still running.
                dev_slot = recv_free_q.get()
                dist.all_gather_into_tensor(self.recv_dev[dev_slot], self.send_dev,
                                            group=topo.group)
                assembled = torch.cuda.Event(blocking=True)
                assembled.record()
                h2d_done.synchronize()
                free_q.put(slot)

                # Hand the window off; the main thread does no host copies and
                # no writes, so it goes straight back to the next collective.
                drain_q.put((dev_slot, round_index, assembled))
        finally:
            # Let the drainer finish the windows already queued before anything
            # is torn down -- it lags the main loop by design.
            drain_q.put(None)
            drain_thread.join()
            # If we left the loop early the reader may be parked on free_q;
            # release it and let it observe the stop flag. filled_q and drain_q
            # are unbounded, so neither thread can block on the other side.
            stop.set()
            for slot in range(SEND_SLOTS):
                free_q.put(slot)
            reader_thread.join()

        if failure:
            raise failure[0]


def build_plan(args):
    """Rank 0 walks the source tree; everyone else takes its answer verbatim."""
    if dist.get_rank() == 0:
        # A planning failure has to be broadcast rather than raised here: every
        # other rank is already waiting in the broadcast below and would hang
        # until the NCCL watchdog fired.
        try:
            jobs = plan_copy_operations(args.sources, args.destination)
            payload = [[(src, dst, size, os.stat(src).st_mtime_ns)
                        for src, dst, size in jobs]]
        except UnreadableEntries as exc:
            payload = [{"error": str(exc)}]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0)
    if isinstance(payload[0], dict):
        raise RuntimeError(f"cannot stage the source tree: {payload[0]['error']}")
    jobs = payload[0]

    # Guard against ranks seeing a different view of shared storage.
    mismatches = 0
    for src, _, size, mtime_ns in jobs:
        try:
            st = os.stat(src)
        except OSError:
            mismatches += 1
            continue
        if st.st_size != size or st.st_mtime_ns != mtime_ns:
            mismatches += 1
    counter = torch.tensor([mismatches], dtype=torch.int64, device="cuda")
    dist.all_reduce(counter)
    if counter.item():
        raise RuntimeError(
            f"{counter.item()} rank/file pairs disagree with rank 0 about the "
            "source tree; shared storage is inconsistent or changing"
        )
    return jobs


def parse_args(argv=None):
    prog = "python3 -m mlperf_common.fileio.datastage"
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Collectively stage SOURCE(s) onto node-local storage on every node.",
        usage=f"""{prog} [OPTION]... SOURCE DEST
       {prog} [OPTION]... SOURCE... DIRECTORY
       {prog} [OPTION]... -t DIRECTORY SOURCE...""",
    )
    parser.add_argument("-t", "--target-directory", metavar="DIRECTORY",
                        help="copy all SOURCE arguments into DIRECTORY")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="copy directories recursively")
    parser.add_argument("files", nargs="+",
                        help="source file(s) and destination (or just sources if -t is used)")
    parser.add_argument("-n", "--num-threads", type=int, default=16,
                        help="reader/writer threads per rank (default: 16). Only helps "
                             "while the per-round read is large; at high node counts the "
                             "window is divided thinly enough that reads are single-threaded.")
    parser.add_argument("-b", "--buffer-size", type=parse_size, default=parse_size("8M"),
                        help="bytes each rank reads per round (default: 8M, must be a "
                             "multiple of 2MiB). The all-gather assembles one of these "
                             "per node into a window that every rank holds, so device "
                             "and pinned host memory scale with the node count: at 2048 "
                             "nodes 8M gives a 16 GiB window.")
    parser.add_argument("--chmod", type=lambda v: int(v, 8), default=0o777,
                        help="octal mode for staged files and directories (default: 0777)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the copy plan and exit")

    args = parser.parse_args(argv)

    if args.target_directory:
        args.sources = args.files
        args.destination = args.target_directory
    else:
        args.sources = args.files[:-1]
        args.destination = args.files[-1]
    if not args.sources:
        parser.error(f"missing destination file operand after '{args.destination}'")
    # O_DIRECT wants alignment, and the all-gather wants a uniform count, so
    # round up rather than rejecting -- same as fastcp.
    if args.buffer_size % BUFFER_ALIGN != 0:
        args.buffer_size = align_up(args.buffer_size, BUFFER_ALIGN)
        print(f"{prog}: rounding buffer size up to {args.buffer_size >> 20} MiB",
              file=sys.stderr)
    # Shared with fastcp, and applied here rather than in build_plan because
    # parse_args runs on every rank before the process group exists: a bad
    # invocation has to kill the whole job uniformly, not leave rank 0 exiting
    # while its peers block in a collective.
    try:
        validate_copy_args(args.sources, args.destination,
                           recursive=args.recursive,
                           into_directory=bool(args.target_directory))
    except CopyArgumentError as exc:
        sys.exit(f"{prog}: {exc}")
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.dry_run and "RANK" not in os.environ:
        try:
            jobs = plan_copy_operations(args.sources, args.destination)
        except UnreadableEntries as exc:
            sys.exit(f"datastage: {exc}")
        for src, dst, size in jobs:
            print(f"{src} -> {dst} ({size} bytes)")
        return 0

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        topology = Topology()
        is_root = topology.rank == 0
        if is_root:
            print(f"datastage: {topology.describe()}", flush=True)

        jobs = build_plan(args)
        total_bytes = sum(size for _, _, size, _ in jobs)
        if is_root:
            print(f"datastage: {len(jobs)} files, {total_bytes / 1e9:.2f} GB", flush=True)
        if args.dry_run:
            if is_root:
                for src, dst, size, _ in jobs:
                    print(f"{src} -> {dst} ({size} bytes)")
            return 0

        stager = Stager(args, topology)
        started = time.monotonic()
        for src, dst, size, mtime_ns in jobs:
            file_started = time.monotonic()
            try:
                stager.stage_file(src, dst, size, mtime_ns)
            except BaseException as exc:  # noqa: BLE001 - annotate then re-raise
                # Peers are blocked in a collective and will be torn down by the
                # NCCL watchdog, so say which rank and which file went wrong.
                print(f"datastage: rank {topology.rank} failed on {src}: {exc}",
                      file=sys.stderr, flush=True)
                raise
            if is_root:
                elapsed = time.monotonic() - file_started
                rate = size / 1e9 / elapsed if elapsed > 0 else 0.0
                print(f"STAGE {src} bytes={size} secs={elapsed:.3f} GB/s={rate:.2f}",
                      flush=True)
        dist.barrier()
        if is_root:
            elapsed = time.monotonic() - started
            rate = total_bytes / 1e9 / elapsed if elapsed > 0 else 0.0
            print(f"DONE files={len(jobs)} secs={elapsed:.3f} GB/s={rate:.2f}", flush=True)
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
