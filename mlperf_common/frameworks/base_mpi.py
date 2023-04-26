# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np

from mlperf_common.frameworks.base import CommunicationHandler

class BaseMPICommunicationHandler(CommunicationHandler):
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
        # c.Barrier()
        val = np.ones(1, dtype=np.int32)
        result = np.zeros(1, dtype=np.int32)
        c.Allreduce(val, result)

    def global_rank(self):
        c = self._get_comm()
        return c.Get_rank()
