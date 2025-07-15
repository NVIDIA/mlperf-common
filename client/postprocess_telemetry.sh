#!/bin/bash

# This script correlates telemetry entries with run_start/run_stop tags removing each entry that is not associated with training time

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <telemetry_log_file> <mllog_file>"
    exit 1
fi

TELEMETRY_FILE=$1
MLLOG_FILE=$2

# 1. look for start/stop tags
# 2. clean the output and extract time_ms from json
# 3. bash magic to make pipe work
read start_ms end_ms < <(
  grep ${MLLOG_FILE} -e "run_start\|run_stop" | \
  sed 's/0: :::MLLOG //' | jq '.time_ms' | \
  sort | head -n 2 | tr '\n' ' '
)

# Date conversion
#export LC_TIME=en_US.UTF-8
#export TZ=America/Los_Angeles
start=$(date -d "@$(($start_ms/1000))" "+%Y/%m/%d %H:%M:%S")
end=$(date -d "@$(($end_ms/1000))" "+%Y/%m/%d %H:%M:%S")


TMP_FILE=$(mktemp)

awk -v start="$start" -v end="$end" '
BEGIN { FS = ", "; OFS = ", " }
NR == 1 { print; next }
{
  timestamp = $1
  if (timestamp >= start && timestamp <= end)
    print
}
' ${TELEMETRY_FILE} > ${TMP_FILE}

mv ${TMP_FILE} ${TELEMETRY_FILE}
