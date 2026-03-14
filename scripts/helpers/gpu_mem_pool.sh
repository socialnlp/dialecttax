#!/usr/bin/env bash
# Memory-budget scheduler for a SINGLE large GPU. Source this, build a JOBS
# array, define a launch callback, and call gpu_mem_pool_run.
#
# The sibling scheduler (gpu_pool.sh) packs jobs by GPU *count*, which is the
# right model for the 8x24GB L4 server: a 12B model needs 8 cards because it
# does not fit on one. On a single 80GB A100 / 180GB B200 that model inverts —
# there is exactly one card, and the scarce resource is its memory. Scheduling
# by count would serialize everything and leave a 1B forward pass using <10% of
# the card. So here each job declares its estimated peak memory and the pool
# packs as many as fit concurrently.
#
# Card capacity is read from nvidia-smi at run time, so the same lane script
# packs 2 jobs on an A100 and 5 on a B200 with no edits.
#
# A job whose estimate exceeds the whole budget (e.g. Llama-70B in bf16 on an
# A100) is not an error: it is dispatched exclusively, and gpu_mem_device tells
# the caller to pass device=auto so accelerate offloads the overflow to host
# RAM. The same job on a B200 fits and gets device=cuda:0.
#
# Peak-memory numbers in the lane scripts are ESTIMATES (weights x a fudge
# factor for grads/activations/KV). If a lane OOMs, raise GPU_MEM_RESERVE_MIB or
# drop GPU_MAX_CONCURRENT rather than re-tuning every entry.
#
# Env knobs:
#   GPU_INDEX            physical GPU to use (default 0)
#   GPU_MEM_RESERVE_MIB  headroom left unallocated (default 8192)
#   GPU_MEM_BUDGET_MIB   override the detected budget entirely
#   GPU_MAX_CONCURRENT   cap on simultaneous jobs (default 4)
#
# Contract for callers:
#   JOBS=( "<peak_mib>|<payload>" ... )   # payload is opaque to the scheduler
#   run_job() { local mib="$1" payload="$2"; ...; }
#   gpu_mem_pool_run run_job
#
# Requires bash >= 5.1 (wait -n -p).

GPU_INDEX="${GPU_INDEX:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

# Peak-memory estimates are approximate and several jobs may share the card, so
# fragmentation is the difference between fitting and an OOM at 99% occupancy.
# expandable_segments lets the allocator grow a segment instead of stranding it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

##############
# DETECTION  #
##############

# Echo "<total_mib> <used_mib>" for GPU_INDEX. Falls back to an idle 80GB card
# when nvidia-smi is unavailable (e.g. --dry-run on a GPU-less box). nvidia-smi
# indexes physical devices, so it is unaffected by CUDA_VISIBLE_DEVICES above.
_gpu_mem_query() {
    local line
    command -v nvidia-smi >/dev/null 2>&1 || { echo "81920 0"; return 0; }
    line=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
           -i "$GPU_INDEX" 2>/dev/null)
    [[ -z "$line" ]] && { echo "81920 0"; return 0; }
    echo "${line//,/ }"
}

# Set GPU_MEM_BUDGET_MIB from the live card unless the caller pinned it.
# --dry-run previews the plan against an assumed-empty card.
_gpu_mem_init() {
    local total used
    read -r total used < <(_gpu_mem_query)
    [[ "${DRY_RUN:-false}" == true ]] && used=0

    # 8 GiB reserve, not 4: each concurrent process pays a ~0.5 GiB CUDA context,
    # and the estimates below are weights x a fudge factor -- a 4 GiB cushion put a
    # 3-job pack at 76000/77824 MiB on an A100 and OOM'd.
    if [[ -z "${GPU_MEM_BUDGET_MIB:-}" ]]; then
        GPU_MEM_BUDGET_MIB=$(( total - used - ${GPU_MEM_RESERVE_MIB:-8192} ))
        (( GPU_MEM_BUDGET_MIB < 1024 )) && GPU_MEM_BUDGET_MIB=1024
    fi
    GPU_MAX_CONCURRENT="${GPU_MAX_CONCURRENT:-4}"

    echo "[$(date '+%H:%M:%S')] GPU $GPU_INDEX: ${total} MiB total, ${used} MiB in use" \
         "-> budget ${GPU_MEM_BUDGET_MIB} MiB, max ${GPU_MAX_CONCURRENT} concurrent"
}

# Echo the device string a job of $1 MiB should load onto: the card if it fits,
# else "auto" so accelerate spills the overflow into host RAM.
gpu_mem_device() {
    (( $1 <= GPU_MEM_BUDGET_MIB )) && echo "cuda:0" || echo "auto"
}

##############
# SCHEDULER  #
##############

# gpu_mem_pool_run <launch_fn>
#   Reads the global JOBS array ("<peak_mib>|<payload>" entries) and packs them
#   onto the single GPU, largest first, until the memory budget or the
#   concurrency cap is reached. <launch_fn> is called as
#   launch_fn "<peak_mib>" "<payload>" and must run the work in the foreground
#   (the scheduler backgrounds it).
#
#   Every job is run to completion; returns 1 if any of them failed, so a lane
#   that OOMs or bails does not report success.
gpu_mem_pool_run() {
    local launch_fn="$1"
    local failed=0
    _gpu_mem_init

    # Largest-first so big jobs claim the card before small ones fragment it.
    # Stable (-s): equal-sized jobs keep the caller's JOBS order.
    local -a pending=(); local _l
    while IFS= read -r _l; do [[ -n "$_l" ]] && pending+=("$_l"); done \
        < <(printf '%s\n' "${JOBS[@]}" | sort -s -t'|' -k1,1nr)

    local -A PIDM=()   # pid -> MiB that pid is holding
    local used=0

    while (( ${#pending[@]} > 0 || ${#PIDM[@]} > 0 )); do
        # 1) Reap finished children and give their memory back.
        local p
        for p in "${!PIDM[@]}"; do
            if ! kill -0 "$p" 2>/dev/null; then
                wait "$p" 2>/dev/null || failed=1
                used=$(( used - PIDM[$p] ))
                echo "[$(date '+%H:%M:%S')] pid=$p done (freed ${PIDM[$p]} MiB, ${used} MiB in flight)"
                unset "PIDM[$p]"
            fi
        done

        # 2) Pack pending jobs into the remaining budget.
        local -a still=(); local placed=0 job
        for job in "${pending[@]}"; do
            local mib="${job%%|*}" payload="${job#*|}"

            if (( mib > GPU_MEM_BUDGET_MIB )); then
                # Does not fit at all: run it alone, offloaded to host RAM.
                if (( ${#PIDM[@]} == 0 && placed == 0 )); then
                    echo "  WARN: job needs ~${mib} MiB > ${GPU_MEM_BUDGET_MIB} MiB budget;" \
                         "running exclusively with CPU offload (device=auto)" >&2
                    "$launch_fn" "$mib" "$payload" &
                    PIDM[$!]=$mib; used=$(( used + mib )); placed=1
                    echo "[$(date '+%H:%M:%S')] dispatch pid=$! (~${mib} MiB, exclusive)"
                else
                    still+=("$job")
                fi
            elif (( ${#PIDM[@]} < GPU_MAX_CONCURRENT && used + mib <= GPU_MEM_BUDGET_MIB )); then
                "$launch_fn" "$mib" "$payload" &
                PIDM[$!]=$mib; used=$(( used + mib )); placed=1
                echo "[$(date '+%H:%M:%S')] dispatch pid=$! (~${mib} MiB, ${used}/${GPU_MEM_BUDGET_MIB} MiB in flight)"
            else
                still+=("$job")
            fi
        done
        pending=("${still[@]}")

        # 3) Block until a job exits (freeing budget) before trying to pack again.
        if (( ${#PIDM[@]} > 0 )); then
            local fp= rc=0
            wait -n -p fp 2>/dev/null || rc=$?
            (( rc != 0 )) && failed=1
            if [[ -n "${fp:-}" && -n "${PIDM[$fp]+x}" ]]; then
                used=$(( used - PIDM[$fp] ))
                echo "[$(date '+%H:%M:%S')] pid=$fp done (freed ${PIDM[$fp]} MiB, ${used} MiB in flight)"
                unset "PIDM[$fp]"
            elif [[ -z "${fp:-}" ]]; then
                PIDM=(); used=0
            fi
        elif (( ${#pending[@]} > 0 )); then
            echo "ERROR: ${#pending[@]} job(s) left but nothing in flight; aborting." >&2
            return 1
        fi
    done

    (( failed )) && { echo "ERROR: one or more jobs failed." >&2; return 1; }
    return 0
}
