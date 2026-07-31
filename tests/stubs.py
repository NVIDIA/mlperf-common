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

"""Enough of torch and torch.distributed to run mlperf_common.fileio on a CPU.

datastage's real dependencies are a GPU and a NCCL job, neither of which is
available where these tests run.  The parts worth testing without them are the
sharding arithmetic, the thread choreography, and whether the bytes that come
out match the bytes that went in -- all of which are independent of the device.

So torch tensors become memoryviews over ctypes buffers, CUDA events become
no-ops, and all_gather_into_tensor becomes whatever the test wants it to be.
What this deliberately does NOT cover: real CUDA events, real NCCL, pinned
memory alignment against a real filesystem block size, and O_DIRECT itself.
"""

import contextlib
import ctypes
import importlib.util
import os
import sys
import threading
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeTensor:
    """A flat byte buffer with the slice/copy_/numpy surface datastage uses.

    Slicing returns a view sharing the parent's memory, as torch does, which is
    what makes the pinned_aligned() offset dance behave the same way here.
    """

    def __init__(self, nbytes=None, buf=None, offset=0, size=None):
        if buf is None:
            buf = ctypes.create_string_buffer(nbytes)
            size = nbytes
        self._buf = buf
        self._offset = offset
        self._size = size

    def _view(self):
        return memoryview(self._buf).cast("B")[self._offset:self._offset + self._size]

    def __len__(self):
        return self._size

    def __getitem__(self, item):
        start, stop, step = item.indices(self._size)
        assert step == 1, "datastage only takes contiguous slices"
        return FakeTensor(buf=self._buf, offset=self._offset + start, size=stop - start)

    def copy_(self, other, non_blocking=False):
        record_op("copy", self)
        source = other._view()
        self._view()[:len(source)] = source
        return self

    def numpy(self):
        return self._view()

    def data_ptr(self):
        return ctypes.addressof(self._buf) + self._offset


class FakeDevice:
    """Stand-in for torch.device('cuda', i).  datastage only ever passes it on."""

    def __init__(self, kind="cuda", index=0):
        self.type = kind
        self.index = index

    def __repr__(self):
        return f"device(type={self.type!r}, index={self.index})"


# CUDA's current device and current stream are both per *host thread*, and a
# thread that never set either gets device 0 and that device's default stream,
# no matter what any other thread did.  Modelling that faithfully is the whole
# point: it is the state a thread doing CUDA work can silently get wrong, and
# no amount of comparing staged bytes on a CPU will show it up.
_CURRENT = threading.local()

# Every event that has been recorded, in order.  test_device.py checks which
# device each one landed on; other tests ignore it.
EVENTS = []

# Every stream-ordered operation, in issue order, tagged with the thread that
# issued it and the stream it went to.  What makes overlap possible is which
# stream work lands on, and that is visible here even though concurrency is
# not.
OPS = []


class Op:
    __slots__ = ("kind", "thread", "stream", "obj")

    def __init__(self, kind, stream, obj):
        self.kind = kind
        self.thread = threading.current_thread().name
        self.stream = stream
        self.obj = obj

    def __repr__(self):
        return f"Op({self.kind}, thread={self.thread!r}, stream={self.stream})"


def record_op(kind, obj=None):
    OPS.append(Op(kind, current_stream(), obj))


class FakeStream:
    """Stand-in for torch.cuda.Stream.

    Only the ordering surface datastage needs: work issued inside a stream
    context belongs to that stream, and wait_event makes this stream wait for
    an event recorded on another one without blocking the host.
    """

    def __init__(self, device=None, default=False):
        self.device = device.index if isinstance(device, FakeDevice) else int(device or 0)
        self.default = default
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)
        OPS.append(Op("wait", self, event))

    def __repr__(self):
        kind = "default" if self.default else "side"
        return f"{kind}-stream(device={self.device})"


_DEFAULT_STREAMS = {}


def current_device():
    return getattr(_CURRENT, "index", 0)


def set_device(device):
    _CURRENT.index = device.index if isinstance(device, FakeDevice) else int(device)


def current_stream(device=None):
    explicit = getattr(_CURRENT, "stream", None)
    if explicit is not None:
        return explicit
    index = current_device() if device is None else device
    return _DEFAULT_STREAMS.setdefault(index, FakeStream(index, default=True))


@contextlib.contextmanager
def stream(target):
    """torch.cuda.stream(): make `target` the calling thread's current stream."""
    previous = getattr(_CURRENT, "stream", None)
    _CURRENT.stream = target
    try:
        yield
    finally:
        _CURRENT.stream = previous


class FakeEvent:
    """CUDA event stand-in: everything is synchronous on the CPU already.

    It does track one thing that is not synchronous, though.  A real
    torch.cuda.Event is created lazily and binds to the *calling thread's*
    current device when it is recorded -- not to the device the work it is
    meant to track ran on.  So an event recorded by a thread that forgot
    set_device belongs to an idle stream on device 0, and synchronize() on it
    returns immediately while the copy it stands for is still in flight.
    Remembering the device here is what lets a test see that.
    """

    def __init__(self, blocking=False):
        self.device = None
        self.stream = None

    def record(self):
        self.device = current_device()
        self.stream = current_stream()
        EVENTS.append(self)
        record_op("record", self)

    def synchronize(self):
        pass


def install(total_memory=288 * 1024 ** 3):
    """Put fake torch / torch.distributed modules into sys.modules."""
    torch = types.ModuleType("torch")
    dist = types.ModuleType("torch.distributed")
    torch.uint8 = "uint8"
    torch.int64 = "int64"
    torch.distributed = dist
    torch.empty = lambda n, dtype=None, device=None, pin_memory=False: FakeTensor(n)
    torch.device = lambda kind="cuda", index=0: FakeDevice(kind, index)
    torch.cuda = types.SimpleNamespace(
        Event=FakeEvent,
        Stream=lambda device=None: FakeStream(device),
        stream=stream,
        current_stream=current_stream,
        current_device=current_device,
        set_device=set_device,
        get_device_properties=lambda device: types.SimpleNamespace(
            total_memory=total_memory),
    )
    dist.barrier = lambda *args, **kwargs: None
    dist.get_rank = lambda: 0
    dist.broadcast_object_list = lambda payload, src=0: None
    dist.all_reduce = lambda tensor: None
    dist.all_gather_into_tensor = lambda out, inp, group=None: out.copy_(inp)
    sys.modules["torch"] = torch
    sys.modules["torch.distributed"] = dist
    return torch, dist


def load_fileio():
    """Import mlperf_common.fileio.* from this checkout, without installing it."""
    package_root = os.path.join(REPO_ROOT, "mlperf_common")
    package = types.ModuleType("mlperf_common")
    package.__path__ = [package_root]
    sys.modules["mlperf_common"] = package
    fileio = types.ModuleType("mlperf_common.fileio")
    fileio.__path__ = [os.path.join(package_root, "fileio")]
    sys.modules["mlperf_common.fileio"] = fileio

    loaded = {}
    for name in ("direct_io", "copyplan", "datastage"):
        spec = importlib.util.spec_from_file_location(
            f"mlperf_common.fileio.{name}",
            os.path.join(package_root, "fileio", f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"mlperf_common.fileio.{name}"] = module
        spec.loader.exec_module(module)
        setattr(fileio, name, module)
        loaded[name] = module
    return loaded
