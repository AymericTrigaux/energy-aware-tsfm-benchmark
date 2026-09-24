#!/usr/bin/env bash
# Is this host quiet enough to trust an energy measurement?
#
# heimdall is shared. CodeCarbon runs in tracking_mode="machine" and the BMC
# power meter reads the whole chassis, so ANY other tenant's load lands in our
# numbers. This is the pre-flight gate.
#
# Exit 0 if 1-minute load average < 8 AND GPU utilisation is 0%; exit 1 otherwise.
#
# Usage:
#   scripts/host_idle_check.sh            # gate
#   scripts/host_idle_check.sh --report   # print only, always exit 0
set -uo pipefail

MAX_LOAD=8.0
REPORT_ONLY=0
[ "${1:-}" = "--report" ] && REPORT_ONLY=1

echo "=============================================================="
echo " Host idle check — $(date -u +%Y-%m-%dT%H:%M:%SZ)  $(hostname)"
echo "=============================================================="

echo
echo "--- uptime / load ---"
uptime

echo
echo "--- logged-in users ---"
who || echo "  (who unavailable)"

echo
echo "--- GPU ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,power.draw,utilization.gpu,memory.used,memory.total \
               --format=csv 2>&1
else
    echo "  nvidia-smi not found"
fi

echo
echo "--- BMC chassis power (10 samples @ 2.0 s) ---"
BMC_DIR=""
for d in /sys/class/hwmon/hwmon*; do
    [ -r "$d/name" ] || continue
    if [ "$(cat "$d/name" 2>/dev/null)" = "power_meter" ]; then BMC_DIR="$d"; break; fi
done
if [ -n "$BMC_DIR" ] && [ -r "$BMC_DIR/power1_average" ]; then
    echo "  sensor: $BMC_DIR/power1_average"
    for i in $(seq 1 10); do
        uw=$(cat "$BMC_DIR/power1_average" 2>/dev/null)
        awk -v i="$i" -v uw="$uw" 'BEGIN{printf "  sample %2d: %8.1f W\n", i, uw/1e6}'
        [ "$i" -lt 10 ] && sleep 2
    done
else
    echo "  no hwmon node named power_meter — BMC power unavailable"
fi

# --- gate ---
LOAD1=$(awk '{print $1}' /proc/loadavg)
GPU_UTIL=0
GPU_UTIL_KNOWN=0
if command -v nvidia-smi >/dev/null 2>&1; then
    # Max utilisation across all GPUs.
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
               | awk 'BEGIN{m=0}{if($1+0>m)m=$1+0}END{print m+0}')
    GPU_UTIL_KNOWN=1
fi

echo
echo "--------------------------------------------------------------"
printf " load average (1 min) : %s   (threshold < %s)\n" "$LOAD1" "$MAX_LOAD"
if [ "$GPU_UTIL_KNOWN" -eq 1 ]; then
    printf " GPU utilisation      : %s%%  (threshold = 0%%)\n" "$GPU_UTIL"
else
    printf " GPU utilisation      : unknown (nvidia-smi absent)\n"
fi
echo "--------------------------------------------------------------"

FAIL=0
REASONS=""
awk -v l="$LOAD1" -v m="$MAX_LOAD" 'BEGIN{exit !(l < m)}' || {
    FAIL=1; REASONS="${REASONS}  - 1-min load average ${LOAD1} >= ${MAX_LOAD}\n"
}
if [ "$GPU_UTIL_KNOWN" -eq 1 ]; then
    [ "$GPU_UTIL" -eq 0 ] || {
        FAIL=1; REASONS="${REASONS}  - GPU utilisation ${GPU_UTIL}% != 0%\n"
    }
else
    FAIL=1; REASONS="${REASONS}  - GPU utilisation could not be determined\n"
fi

if [ "$REPORT_ONLY" -eq 1 ]; then
    echo "REPORT ONLY — not gating (exit 0)"
    exit 0
fi

if [ "$FAIL" -eq 0 ]; then
    echo "PASS — host is quiet enough for an energy measurement."
    exit 0
fi

echo "FAIL — host is NOT quiet; energy numbers would include other tenants' load:"
printf "%b" "$REASONS"
echo
echo "Wait for the host to settle, or re-run with --report to inspect without gating."
exit 1
