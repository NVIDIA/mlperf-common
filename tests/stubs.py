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

import ctypes
import importlib.util
import os
import sys
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
        source = other._view()
        self._view()[:len(source)] = source
        return self

    def numpy(self):
        return self._view()

    def data_ptr(self):
        return ctypes.addressof(self._buf) + self._offset


class FakeEvent:
    """CUDA event stand-in: everything is synchronous on the CPU already."""

    def __init__(self, blocking=False):
        pass

    def record(self):
        pass

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
    torch.device = lambda *args, **kwargs: "cpu"
    torch.cuda = types.SimpleNamespace(
        Event=FakeEvent,
        current_device=lambda: 0,
        set_device=lambda device: None,
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
