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

import mxnet as mx
from mxnet import cuda_utils as cu
import numpy as np

from mlperf_common.frameworks.base import CommunicationHandler, ProfilerHandler

class MPICommunicationHandler(CommunicationHandler):
    def __init__(self, comm=None, **kwargs):
        super().__init__(**kwargs)
        self.comm = comm

    def _get_comm(self):
        if self.comm is None:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD

        return self.comm

    def barrier(self, sync_group=None):
        c = self._get_comm() if sync_group is None else sync_group
        # NOTE: MPI_Barrier is *not* working reliably at scale. Using MPI_Allreduce instead.
        #c.Barrier()
        val = np.ones(1, dtype=np.int32)
        result = np.zeros(1, dtype=np.int32)
        c.Allreduce(val, result)

    def global_rank(self):
        c = self._get_comm()
        return c.Get_rank()

    def device_sync(self):
        mx.nd.waitall()
        
    def allreduce(self, x):
        val = np.array(x, dtype=np.int32)
        result = np.zeros_like(val, dtype=np.int32)
        self._get_comm().Allreduce([val, self.MPI.INT], [result, self.MPI.INT])
        return result


class MXNetProfilerHandler(ProfilerHandler):
    def profiler_start(self):
        cu.cuda_profiler_start()

    def profiler_stop(self):
        cu.cuda_profiler_stop()

    def push_nvtx(self, tag):
        cu.nvtx_range_push(tag)

    def pop_nvtx(self):
        cu.nvtx_range_pop()


