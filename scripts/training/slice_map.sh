#!/bin/bash
# Host: azkaban — 1× NVIDIA RTX PRO 6000 Blackwell, MIG-partitioned into 4× 1g.24gb slices.
# 128 logical CPUs, 2 sockets × 32 cores × 2 threads. NUMA: cores interleaved
# (even=node0, odd=node1), so any contiguous 32-core range is NUMA-balanced.
#
# This file maps slice index (0..3) → CUDA UUID + a non-overlapping CPU range.
# Source it from any script that launches training:
#
#   source scripts/training/slice_map.sh
#   SLICE=0
#   CUDA_VISIBLE_DEVICES=$(slice_cuda_uuid $SLICE) \
#   TASKSET_CPUS=$(slice_cpu_range $SLICE) \
#   ... auto_restart.sh
#
# If MIG is reconfigured, regenerate by running `nvidia-smi -L` and editing below.

NPROC="${NPROC_OVERRIDE:-$(nproc)}"
SLICE_COUNT=4
PER_SLICE=$(( NPROC / SLICE_COUNT ))

# MIG UUIDs as listed by `nvidia-smi -L` on azkaban (2026-04-25 partitioning).
declare -A _MIG_UUIDS=(
    [0]="MIG-5dc2f940-5003-58b0-a068-bede55f1d56f"
    [1]="MIG-abbb3894-4f8c-5e33-b602-6a485436950d"
    [2]="MIG-dd607cdf-e8cb-531f-b478-417160625a35"
    [3]="MIG-7488039b-0c78-50bb-8112-a1ae051fc3f7"
)

slice_cuda_uuid() {
    local idx=$1
    local uuid="${_MIG_UUIDS[$idx]:-}"
    if [ -z "$uuid" ]; then
        echo "slice_map: no MIG UUID for slice index $idx (valid: 0-3)" >&2
        return 1
    fi
    echo "$uuid"
}

slice_cpu_range() {
    local idx=$1
    if [ "$idx" -lt 0 ] || [ "$idx" -ge "$SLICE_COUNT" ]; then
        echo "slice_map: invalid slice index $idx (valid: 0-$((SLICE_COUNT-1)))" >&2
        return 1
    fi
    local lo=$(( idx * PER_SLICE ))
    local hi=$(( (idx + 1) * PER_SLICE - 1 ))
    echo "${lo}-${hi}"
}

# Print the mapping when this file is run directly (for human inspection).
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Host CPUs: $NPROC, slices: $SLICE_COUNT, cores/slice: $PER_SLICE"
    for i in 0 1 2 3; do
        printf "  slice %d  cuda=%s  cpus=%s\n" "$i" "$(slice_cuda_uuid $i)" "$(slice_cpu_range $i)"
    done
fi
