#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

set -euo pipefail

[[ "${DEBUG:-}" ]] && set -x

SCRIPT_NAME=$(basename "${0}")
USAGE_STRING="\
usage: ${SCRIPT_NAME}

"


# variables we derive from the NVIDIA container
: "${NVIDIA_PRODUCT_NAME:=FIXME?UNKNOWN}"
: "${CUDA_VERSION:=FIXME?UNKNOWN}"
: "${CUDA_DRIVER_VERSION:=FIXME?UNKNOWN}"
: "${NCCL_VERSION:=FIXME?UNKNOWN}"
: "${CUBLAS_VERSION:=FIXME?UNKNOWN}"
: "${CUDNN_VERSION:=FIXME?UNKNOWN}"
: "${TRT_VERSION:=FIXME?UNKNOWN}"
: "${DALI_VERSION:=FIXME?UNKNOWN}"
: "${MOFED_VERSION:=FIXME?UNKNOWN}"
: "${OPENMPI_VERSION:=FIXME?UNKNOWN}"


NVIDIA_VERSION_VARIABLE_NAME="NVIDIA_${NVIDIA_PRODUCT_NAME^^}_VERSION"
NVIDIA_PRODUCT_VERSION="${!NVIDIA_VERSION_VARIABLE_NAME}"
NVIDIA_PRODUCT_VERSION_SHORT="$(sed -E 's/^(\d\d\.\d\d).*$/\1/' <<< ${NVIDIA_PRODUCT_VERSION})"

# variables we require, any of these can be set from outside the script
: "${MLPERF_SUBMITTER:=FIXME?UNSPECIFIED_SUBMITTER}"

# these variables can be set outside the script
# division: "closed" or "open"
: "${MLPERF_DIVISION:=FIXME?UNSPECIFIED_DIVISION}"
# status: is called "category" in the rules, and I think the result summarizer
# calls it "availability" I am unclear about what the expected values are.
# We've been using onprem for "available on premises" and "preview" for preview
: "${MLPERF_STATUS:=FIXME?UNSPECIFIED_STATUS}"
: "${MLPERF_NUM_NODES:=${DGXNNODES:-${SLURM_JOB_NUM_NODES:-FIXME?UNKNOWN}}}"

# need to define MLPERF_SYSTEM_NAME
# it can be defined in the environment that calls this script
# else we search for a helper script
# that calls this script, if there's a helper script available, then use that:
if [[ -x "./${MLPERF_SUBMITTER}-system-name.sh" ]]; then
    source "./${MLPERF_SUBMITTER}-system-name.sh"
fi
# make sure MLPERF_SYSTEM_NAME is set to something
: "${MLPERF_SYSTEM_NAME:=FIXME?UNSPECIFIED_SYSTEM_NAME}"

# FIXME: this is NVIDIA specific naming convention, so not appropriate for anyone
# other than NVIDIA (i.e., this is what should go into NVIDIA-system-name.sh)
if [[ "${MLPERF_NUM_NODES}" -gt "1" ]]; then
    MLPERF_SYSTEM_NAME="dgxh100_n${MLPERF_NUM_NODES}_ngc${NVIDIA_PRODUCT_VERSION_SHORT}_${NVIDIA_PRODUCT_NAME,,}"
else
    # we special case our system name for single node:
    MLPERF_SYSTEM_NAME="dgxh100_ngc${NVIDIA_PRODUCT_VERSION_SHORT}_${NVIDIA_PRODUCT_NAME,,}"
fi


# variables we get from the OS environment inside the container
function get_lscpu_info() {
    lscpu --json | python -c "\
import json, sys; \
print(next( \
    item['data'] \
    for item in json.load(sys.stdin)['lscpu'] \
        if item['field'] == '$1' \
    ))"
}

: "${NVIDIA_KERNEL_DRIVER:=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0)}"
: "${NV_ACC_NAME:=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader --id=0)}"
: "${MLPERF_ACC_NUM:=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc --lines)}"
: "${MLPERF_OS_PRETTY_NAME:=$(source <(cat /etc/os-release | grep "^PRETTY_NAME=") && echo "${PRETTY_NAME}")}"
: "${MLPERF_CPU_SOCKETS:=$(get_lscpu_info "Socket(s):")}"
: "${MLPERF_CORES_PER_SOCKET:=$(get_lscpu_info "Core(s) per socket:")}"
: "${MLPERF_CPU_MODEL:=$(get_lscpu_info 'Model name:')}"
: "${MLPERF_HOST_MEM:=$(awk '/MemTotal/{printf "%.1f TB\n", $2/(1024*1024*1024)}' /proc/meminfo)}"
: "${MLPERF_LINUX_KERNEL_RELEASE:=$(uname --kernel-name --kernel-release)}"



OUTPUT_STRING=$(cat <<EOF
{
    "submitter": "${MLPERF_SUBMITTER}",
    "division": "${MLPERF_DIVISION}",
    "status": "${MLPERF_STATUS}",
    "system_name": "${MLPERF_SYSTEM_NAME}",
    "number_of_nodes": "${MLPERF_NUM_NODES}",
    "host_processors_per_node": "${MLPERF_CPU_SOCKETS}",
    "host_processor_model_name": "${MLPERF_CPU_MODEL}",
    "host_processor_core_count": "${MLPERF_CORES_PER_SOCKET}",
    "host_processor_vcpu_count": "",
    "host_processor_frequency": "",
    "host_processor_caches": "",
    "host_processor_interconnect": "",
    "host_memory_capacity": "${MLPERF_HOST_MEM}",
    "host_storage_type": "NVMe SSD",
    "host_storage_capacity": "2x 1.92TB NVMe SSD + 30TB U.2 NVMe SSD",
    "host_networking": "Storage: 2x ConnectX-7 IB NDR 400Gb/s, Compute: 8x ConnectX-7 IB NDR 400Gb/s, Management: 100Gb/s Ethernet NIC",
    "host_networking_topology": "",
    "host_memory_configuration": "",
    "accelerators_per_node": "${MLPERF_ACC_NUM}",
    "accelerator_model_name": "${NV_ACC_NAME}",
    "accelerator_host_interconnect": "",
    "accelerator_frequency": "",
    "accelerator_on-chip_memories": "",
    "accelerator_memory_configuration": "HBM3",
    "accelerator_memory_capacity": "80 GB",
    "accelerator_interconnect": "NVLINK Gen4 900 GB/s + NVSWITCH Gen3",
    "accelerator_interconnect_topology": "",
    "cooling": "",
    "hw_notes": "",
    "framework": "${NVIDIA_PRODUCT_NAME} NVIDIA Release ${NVIDIA_PRODUCT_VERSION}",
    "other_software_stack": {
        "cuda_version": "${CUDA_VERSION}",
        "cuda_driver_version": "${CUDA_DRIVER_VERSION}",
        "nccl_version": "${NCCL_VERSION}",
        "cublas_version": "${CUBLAS_VERSION}",
        "cudnn_version": "${CUDNN_VERSION}",
        "trt_version": "${TRT_VERSION}",
        "dali_version": "${DALI_VERSION}",
        "mofed_version": "${MOFED_VERSION}",
        "openmpi_version": "${OPENMPI_VERSION}",
	"kernel_version": "${MLPERF_LINUX_KERNEL_RELEASE}",
	"nvidia_kernel_driver": "${NVIDIA_KERNEL_DRIVER}"
    },
    "operating_system": "${MLPERF_OS_PRETTY_NAME}",
    "sw_notes": ""
}
EOF
		)

echo ${OUTPUT_STRING} | python3 -m json.tool

