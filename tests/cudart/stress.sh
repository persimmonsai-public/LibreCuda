#!/bin/bash
# Runs the CUDA runtime test suite repeatedly and checks the GPU's PCIe error status (AER) and the kernel log for Xids
# after every run. Stops at the first Xid, since the driver is usually wedged afterwards.
#
# usage: stress.sh <build dir> <runs> [GPU PCI address] [log dir]
#   The PCI address (e.g. 02:00.0) enables the AER checks, which need sudo for setpci / lspci.
#   CUDA_VISIBLE_DEVICES, LIBRECUDA_WAIT_TIMEOUT_MS etc. are passed through to the tests.
set -u
script_dir="$(cd "$(dirname "$0")" && pwd)"
build_dir="$1"
runs="$2"
pci="${3:-}"
log_dir="${4:-$build_dir/tests/cudart/stress}"
mkdir -p "$log_dir"

xid_count() { journalctl -k -b --no-pager | grep -cE "Xid|Heartbeat"; }

xids_before=$(xid_count)
failed_runs=0
for i in $(seq 1 "$runs"); do
    if [ -n "$pci" ]; then
        # the AER status registers are write-1-to-clear
        sudo setpci -s "$pci" ECAP_AER+0x04.L=0xffffffff
        sudo setpci -s "$pci" ECAP_AER+0x10.L=0xffffffff
    fi
    start=$(date +%s)
    "$script_dir/run_tests.sh" "$build_dir" > "$log_dir/run$i.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && failed_runs=$((failed_runs + 1))
    aer=""
    if [ -n "$pci" ]; then
        aer=" aer=[$(sudo lspci -vv -s "$pci" | grep -E "UESta|CESta" | grep -oE "[A-Za-z]+\+" | tr '\n' ' ')]"
    fi
    new_xids=$(($(xid_count) - xids_before))
    echo "run $i rc=$rc $(($(date +%s) - start))s$aer new_xids=$new_xids :: $(tail -1 "$log_dir/run$i.log")"
    if [ $new_xids -gt 0 ]; then
        echo "Xid logged, stopping"
        journalctl -k -b --no-pager | grep -E "Xid|Heartbeat" | tail -5
        exit 2
    fi
done
echo "$((runs - failed_runs)) of $runs runs passed"
[ $failed_runs -eq 0 ]
