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

"""Fast file I/O: O_DIRECT primitives, copy planning, and collective staging.

  direct_io   O_DIRECT pread/pwrite with aligned buffers (fastcp, fastmd5, datastage)
  copyplan    source-tree walk and src->dst mapping (fastcp, datastage)
  datastage   NCCL-collective dataset staging onto node-local storage

Only datastage requires torch; the other two are dependency-free so that the
single-node client scripts do not pull in a training stack.
"""
