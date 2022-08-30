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

class CommunicationHandler:
    def global_rank(self):
        raise NotImplementedError

    def barrier(self, sync_group=None):
        raise NotImplementedError

    def device_sync(self):
        raise NotImplementedError

class ProfilerHandler:
    def profiler_start(self):
        raise NotImplementedError

    def profiler_stop(self):
        raise NotImplementedError

    def push_nvtx(self, tag):
        raise NotImplementedError

    def pop_nvtx(self):
        raise NotImplementedError

