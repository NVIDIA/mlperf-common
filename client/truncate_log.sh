#!/bin/bash

# Usage: truncate_log.sh <size_threshold_mb> <log_files...>
# Example: truncate_log.sh 10 /results/log1.log /results/log2.log

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <size_threshold_mb> <log_files...>"
    exit 1
fi

SIZE_THRESHOLD_MB=$1
shift
SIZE_IN_BYTES=$((SIZE_THRESHOLD_MB * 1024 * 1024))
HEAD_SIZE=$(( SIZE_IN_BYTES * 9 / 10 ))
TAIL_SIZE=$(( SIZE_IN_BYTES * 1 / 10 ))

for log_file in "$@"; do
    if [[ ! -f "${log_file}" ]]; then
        continue
    fi
    
    LOG_SIZE_MB=$(( $(stat -c %s "${log_file}" 2>/dev/null) / 1024 / 1024 ))
    
    if [ ${LOG_SIZE_MB} -gt ${SIZE_THRESHOLD_MB} ]; then
        # Save original file
        mv ${log_file} ${log_file}.original
        
        # Calculate how many MB will be removed for the message
        REMOVED_MB=$((LOG_SIZE_MB - SIZE_THRESHOLD_MB))
        MESSAGE="========== REMOVED ${REMOVED_MB}MB OF LOG =========="
        
        # Truncate the file with head and tail
        (head -c ${HEAD_SIZE} ${log_file}.original; 
         echo ${MESSAGE}; 
         tail -c ${TAIL_SIZE} ${log_file}.original) > ${log_file}
    fi
done
