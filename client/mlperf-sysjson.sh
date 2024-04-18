#!/bin/bash

# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
   behavior is controlled by envvars
   Required:
   * MLPERF_SUBMITTER
   * MLPERF_SYSTEM_NAME
   * MLPERF_STATUS (must be 'onprem', 'cloud', 'preview', or 'research')

   Required but usually have reasonable defaults:
   * MLPERF_DIVISION (defaults to 'closed', may change to 'open')
   * MLPERF_NUM_NODES (defaults to DGXNNODES if defined)

   Optional:
    * MLPERF_HOST_STORAGE_TYPE
    * MLPERF_HOST_STORAGE_CAPACITY
    * MLPERF_HOST_NETWORKING
    * MLPERF_HOST_NETWORKING_TOPOLOGY
    * MLPERF_HOST_MEMORY_CONFIGURATION
    * MLPERF_ACCELERATOR_MODEL_NAME
    * MLPERF_ACCELERATOR_HOST_INTERCONNECT
    * MLPERF_ACCELERATOR_FREQUENCY
    * MLPERF_ACCELERATOR_ON_CHIP_MEMORIES
    * MLPERF_ACCELERATOR_MEMORY_CONFIGURATION
    * MLPERF_ACCELERATOR_INTERCONNECT
    * MLPERF_ACCELERATOR_INTERCONNECT_TOPOLOGY
    * MLPERF_COOLING
    * MLPERF_HW_NOTES

    Automatically generated:
    * most of the rest of the fields in the system json, including things like
      * cpu sockets, cores, model name
      * accelerator model name, quantity
      * cuda and library versions
"
###############################################################################
# variables we require, these must be set from outside the script since we
# don't have any way to calculate a reasonable default value
###############################################################################
: "${MLPERF_SUBMITTER:="UNKNOWN_MLPERF_SUBMITTER"}"

# status: is called "category" in the rules, and the result summarizer calls it
# "availability"
# https://github.com/mlcommons/policies/blob/master/submission_rules.adoc#73-results-categories
# specifies 4 "categories": "Available in cloud", "Available on premise",
# "Preview", and "Research, Development, or Internal", but
# https://github.com/mlcommons/policies/blob/master/submission_rules.adoc#57-system_desc_idjson-metadata
# does not specify what the corresponding strings should be in the "status"
# field of system_desc_id.json.  In the past some people have used "available",
# "onprem", "cloud", "preview", "research"
# in 3.1 round it was clarified that 
: "${MLPERF_STATUS:="UNKNOWN_MLPERF_STATUS"}"

: "${MLPERF_SYSTEM_NAME:="UNKNOWN_MLPERF_SYSTEM_NAME"}"

###############################################################################
# variables we require, but for which we can sometimes generate a reasonable
# default.  This may be overridden from outside the script if the default isn't
# appropriate.
###############################################################################
: "${MLPERF_NUM_NODES:=${DGXNNODES:-${SLURM_JOB_NUM_NODES:-"1"}}}"

# division: "closed" by default, may be overridden to "open"
: "${MLPERF_DIVISION:=closed}"

# correctness check for division
if [[ ! ( "${MLPERF_DIVISION}" == "closed" || "${MLPERF_DIVISION}" == "open" ) ]]; then
    echo "the only legal values for MLPERF_DIVISION are 'closed' or 'open'" 1>&2
    echo
    echo "${USAGE_STRING}"
    exit 1
fi

# correctness check for status
case "${MLPERF_STATUS}" in
    "onprem"|"cloud"|"preview"|"research")
	true ;;
    *)
	echo "the only legal values for MLPERF_STATUS are" 1>&2
	echo "* onprem (means: available on premise)" 1>&2
	echo "* cloud  (means: available in cloud)"   1>&2
	echo "* preview" 1>&2
	echo "* reserach (means: research, devlopment, or internal)" 1>&2
	echo
	echo "${USAGE_STRING}"
	exit 1
	;;
esac

###############################################################################
# optional variables, these may be set from outside the script
###############################################################################
: "${MLPERF_FRAMEWORK:=""}"
: "${MLPERF_FRAMEWORK_SHORT_NAME:=""}"
: "${MLPERF_HOST_STORAGE_TYPE:=""}"
: "${MLPERF_HOST_STORAGE_CAPACITY:=""}"
: "${MLPERF_HOST_NETWORKING:=""}"
: "${MLPERF_HOST_NETWORKING_TOPOLOGY:=""}"
: "${MLPERF_HOST_MEMORY_CONFIGURATION:=""}"
: "${MLPERF_ACCELERATOR_HOST_INTERCONNECT:=""}"
: "${MLPERF_ACCELERATOR_FREQUENCY:=""}"
: "${MLPERF_ACCELERATOR_ON_CHIP_MEMORIES:=""}"
: "${MLPERF_ACCELERATOR_MEMORY_CONFIGURATION:=""}"
: "${MLPERF_ACCELERATOR_INTERCONNECT:=""}"
: "${MLPERF_ACCELERATOR_INTERCONNECT_TOPOLOGY:=""}"
: "${MLPERF_COOLING:=""}"
: "${MLPERF_HW_NOTES:=""}"

# if caller defines MLPERF_SYSJSON_SYSNAME_INCLUDE_NUM_NODES we construct a
# more interesting system name for multi-node systems
if [[ "${MLPERF_SYSJSON_SYSNAME_INCLUDE_NUM_NODES:-}" ]] && ((MLPERF_NUM_NODES > 1)); then
    MLPERF_SYSTEM_NAME="${MLPERF_SYSTEM_NAME}_n${MLPERF_NUM_NODES}"
fi

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
NVIDIA_PRODUCT_VERSION_SHORT="$(sed -E 's/^(\d\d\.\d\d).*$/\1/' <<< "${NVIDIA_PRODUCT_VERSION}")"



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
: "${MLPERF_ACCELERATOR_MODEL_NAME:=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader --id=0)}"
: "${MLPERF_ACC_NUM:=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc --lines)}"
: "${MLPERF_ACC_MEM:=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader --id=0)}"
: "${MLPERF_OS_PRETTY_NAME:=$(sed -En -e 's/^PRETTY_NAME="([^"]*)"$/\1/p' /etc/os-release)}"
: "${MLPERF_CPU_SOCKETS:=$(get_lscpu_info "Socket(s):")}"
: "${MLPERF_CORES_PER_SOCKET:=$(get_lscpu_info "Core(s) per socket:")}"
: "${MLPERF_CPU_MODEL:=$(get_lscpu_info 'Model name:')}"
: "${MLPERF_HOST_MEM:=$(awk '/MemTotal/{printf "%.1f TB\n", $2/(1024*1024*1024)}' /proc/meminfo)}"
: "${MLPERF_LINUX_KERNEL_RELEASE:=$(uname --kernel-name --kernel-release)}"
: "${MLPERF_FRAMEWORK:="${NVIDIA_PRODUCT_NAME} NVIDIA Release ${NVIDIA_PRODUCT_VERSION}"}"



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
    "host_storage_type": "${MLPERF_HOST_STORAGE_TYPE:-}",
    "host_storage_capacity": "${MLPERF_HOST_STORAGE_CAPACITY:-}",
    "host_networking": "${MLPERF_HOST_NETWORKING:-}",
    "host_networking_topology": "${MLPERF_HOST_NETWORKING_TOPOLOGY:-}",
    "host_memory_configuration": "${MLPERF_HOST_MEMORY_CONFIGURATION:-}",
    "accelerators_per_node": "${MLPERF_ACC_NUM}",
    "accelerator_model_name": "${MLPERF_ACCELERATOR_MODEL_NAME}",
    "accelerator_host_interconnect": "${MLPERF_ACCELERATOR_HOST_INTERCONNECT:-}",
    "accelerator_frequency": "${MLPERF_ACCELERATOR_FREQUENCY:-}",
    "accelerator_on-chip_memories": "${MLPERF_ACCELERATOR_ON_CHIP_MEMORIES:-}",
    "accelerator_memory_configuration": "${MLPERF_ACCELERATOR_MEMORY_CONFIGURATION:-}",
    "accelerator_memory_capacity": "${MLPERF_ACC_MEM:-}",
    "accelerator_interconnect": "${MLPERF_ACCELERATOR_INTERCONNECT:-}",
    "accelerator_interconnect_topology": "${MLPERF_ACCELERATOR_INTERCONNECT_TOPOLOGY:-}",
    "cooling": "${MLPERF_COOLING:-}",
    "hw_notes": "${MLPERF_HW_NOTES:-}",
    "framework": "${MLPERF_FRAMEWORK}",
    "framework_name": "${MLPERF_FRAMEWORK_SHORT_NAME:-}",
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

echo "${OUTPUT_STRING}" | python3 -m json.tool --compact
