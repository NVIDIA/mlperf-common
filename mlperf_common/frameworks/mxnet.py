# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

import mxnet as mx
from mxnet import cuda_utils as cu

from mlperf_common.frameworks.base import BaseMPICommunicationHandler, ProfilerHandler

class MPICommunicationHandler(BaseMPICommunicationHandler):
    def device_sync(self):
        mx.nd.waitall()

class MXNetProfilerHandler(ProfilerHandler):
    def profiler_start(self):
        cu.cuda_profiler_start()

    def profiler_stop(self):
        cu.cuda_profiler_stop()

    def push_nvtx(self, tag):
        cu.nvtx_range_push(tag)

    def pop_nvtx(self):
        cu.nvtx_range_pop()
