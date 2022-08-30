# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import torch

from mlperf_common.frameworks.base import CommunicationHandler, ProfilerHandler

class PyTCommunicationHandler(CommunicationHandler):
    def global_rank(self):
        if self.is_distributed():
            return torch.distributed.get_rank()
        return 0

    def barrier(self, sync_group=None):
        if self.is_distributed():
            torch.distributed.barrier(sync_group)

    def is_distributed(self):
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def device_sync(self):
        torch.cuda.synchronize()

class PyTProfilerHandler(ProfilerHandler):
    def profiler_start(self):
        torch.cuda.cudart().cudaProfilerStart()

    def profiler_stop():
        torch.cuda.cudart().cudaProfilerStop()

    def push_nvtx(self, tag):
        torch.cuda.nvtx.range_push(tag)

    def pop_nvtx(self):
        torch.cuda.nvtx.range_pop()

