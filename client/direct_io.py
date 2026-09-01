#!/usr/bin/env python3

# Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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

# Compatibility shim.  direct_io now lives in the mlperf_common package so that
# it can be imported by package modules (mlperf_common.fileio.datastage) and not only
# by the scripts installed alongside it in bin/.  This shim keeps `import
# direct_io` working for anything that still expects a module next to fastcp.

# The package is found either because mlperf-common is installed (setup.py
# installs the package and these scripts together, so this holds for any pip
# install) or because we are running from a tree with mlperf_common/ next to
# client/ -- which covers the source checkout and the "deploy the repo to a
# shared filesystem and run client/ scripts in place" pattern.  Copying
# individual scripts out of client/ on their own does not work.

import os
import sys

try:
    from mlperf_common.fileio.direct_io import *  # noqa: F401,F403
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    try:
        from mlperf_common.fileio.direct_io import *  # noqa: F401,F403
    except ImportError as exc:
        raise ImportError(
            f"{exc}. direct_io lives in the mlperf_common package. Either install "
            "mlperf-common, or keep this script in a tree with mlperf_common/ "
            "alongside client/."
        ) from exc
