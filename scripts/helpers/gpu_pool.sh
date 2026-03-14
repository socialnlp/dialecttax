#!/usr/bin/env bash
# Shared GPU-pool scheduler for the generation lanes. Source this, build a JOBS
# array, define a launch callback, and call gpu_pool_run.
#
# Instead of hardcoding GPU indices (0..7) or killing every GPU process on the
# box, the scheduler discovers *free* GPUs from nvidia-smi and only runs on
# those, so a lane coexists with other jobs / sibling lanes. It re-polls
# continuously: as GPUs free up (whether from our own finishing jobs or an
# unrelated process), queued work moves onto them.
#
# Free = a GPU whose used memory is below GPU_FREE_MEM_MIB (default 1024 MiB).
#
# Cross-process reservation (flock): two lanes launched at the same instant would
# both see an idle GPU as "free" before either's job has claimed memory. To close
# that race, each GPU is flock'd (advisory lock under GPU_LOCK_DIR) before launch;
# a GPU another lane already holds fails to lock and is skipped. The lock is held
# by the lane process for the job's lifetime and released by the kernel the moment
# the lane exits (even on SIGKILL), so there are no stale reservations to clean
# up. Lock dir is per-user by default, so a user's own lanes coordinate; other
# users' jobs are still avoided via their nvidia-smi memory once loaded. Degrades
# to in-process-only scheduling if flock is unavailable or GPU_LOCK_DIR is not
# writable.
#
# Env knobs:
#   GPU_FREE_MEM_MIB   used-memory threshold for "free" (default 1024)
#   GPU_POLL_SECS      re-poll interval when the queue is blocked (default 15)
#   GPU_POOL           restrict to these indices (space/comma list); freeness is
#                      still checked within them. Unset = consider all GPUs.
#   GPU_LOCK_DIR       reservation lock dir (default /tmp/dialecttax-gpu-locks-$UID)
#
# Contract for callers:
#   JOBS=( "<n_gpus>|<payload>" ... )   # payload is opaque to the scheduler
#   run_job() { local gpus="$1" payload="$2"; ...; }   # gpus is comma-joined
#   gpu_pool_run run_job
#
# Requires bash >= 5.1 (wait -n -p).

##############
# DETECTION  #
##############

# Echo the space-separated indices of GPUs that are currently free (used memory
# < threshold). Restricted to GPU_POOL when that is set. Falls back to 0..7 when
# nvidia-smi is unavailable (e.g. --dry-run on a GPU-less box).
gpu_nvidia_free() {
    local thresh="${1:-${GPU_FREE_MEM_MIB:-1024}}"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "${GPU_POOL:-0 1 2 3 4 5 6 7}" | tr ',' ' '
        return 0
    fi
    local -A allowed=(); local restricted=0 g
    if [[ -n "${GPU_POOL:-}" ]]; then
        restricted=1
        for g in ${GPU_POOL//,/ }; do allowed[$g]=1; done
    fi
    local idx used free=()
    while IFS=',' read -r idx used; do
        idx="${idx//[[:space:]]/}"; used="${used//[[:space:]]/}"
        [[ -z "$idx" ]] && continue
        (( restricted )) && [[ -z "${allowed[$idx]:-}" ]] && continue
        (( used < thresh )) && free+=("$idx")
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
    echo "${free[*]:-}"
}

##################
# RESERVATION    #
##################

GPU_LOCK_DIR="${GPU_LOCK_DIR:-/tmp/dialecttax-gpu-locks-$UID}"
declare -A _GPU_FD=()   # gpu idx -> open fd holding that GPU's flock (lane-held)
_GPU_HAVE_FLOCK=0
if command -v flock >/dev/null 2>&1 && mkdir -p "$GPU_LOCK_DIR" 2>/dev/null && [[ -w "$GPU_LOCK_DIR" ]]; then
    _GPU_HAVE_FLOCK=1
fi

# Atomically reserve GPU $1 across processes. On success, records the holding fd
# in _GPU_FD[$1] and returns 0; returns 1 if another lane already holds it.
_gpu_reserve() {
    local idx="$1" fd
    exec {fd}>"$GPU_LOCK_DIR/gpu_$idx.lock" 2>/dev/null || return 1
    if flock -n "$fd"; then
        _GPU_FD[$idx]="$fd"
        return 0
    fi
    eval "exec ${fd}>&-" 2>/dev/null || true
    return 1
}

# Release our flocks on the given GPU indices (space-separated args).
_gpu_release() {
    local idx fd
    for idx in "$@"; do
        fd="${_GPU_FD[$idx]:-}"
        [[ -z "$fd" ]] && continue
        eval "exec ${fd}>&-" 2>/dev/null || true
        unset "_GPU_FD[$idx]"
    done
}

##############
# SCHEDULER  #
##############

# gpu_pool_run <launch_fn>
#   Reads the global JOBS array ("<n_gpus>|<payload>" entries) and dispatches
#   each onto free GPUs, largest job first, re-polling until all are done.
#   <launch_fn> is called as: launch_fn "<comma,gpus>" "<payload>" and must
#   start the work in the foreground (the scheduler backgrounds it).
gpu_pool_run() {
    local launch_fn="$1"
    local poll="${GPU_POLL_SECS:-15}"

    # --dry-run previews the plan against an assumed-empty machine (a huge
    # threshold makes every GPU count as free) and skips cross-process locking,
    # so it never stalls on live state or touches the lock dir.
    local thresh="${GPU_FREE_MEM_MIB:-1024}" use_locks="$_GPU_HAVE_FLOCK"
    [[ "${DRY_RUN:-false}" == true ]] && { thresh=99999999; poll=1; use_locks=0; }

    # Queue, sorted largest-first so multi-GPU jobs schedule before 1-GPU fill.
    # Stable (-s): jobs of equal GPU count keep the caller's JOBS order rather
    # than being reordered alphabetically by the line contents.
    local -a pending=(); local _l
    while IFS= read -r _l; do [[ -n "$_l" ]] && pending+=("$_l"); done \
        < <(printf '%s\n' "${JOBS[@]}" | sort -s -t'|' -k1,1nr)

    # Total GPUs in our universe (all visible / GPU_POOL-restricted), used only
    # to clamp any job that asks for more than can ever be free.
    local total; total=$(gpu_nvidia_free 999999999 | wc -w); (( total == 0 )) && total=8

    local -A PIDG=()       # pid -> comma-joined GPUs the job holds
    local -A reserved=()   # gpu -> 1 for GPUs our in-flight jobs hold, rebuilt each round

    while (( ${#pending[@]} > 0 || ${#PIDG[@]} > 0 )); do
        # 1) Reap finished children; release their GPU locks.
        local p
        for p in "${!PIDG[@]}"; do
            if ! kill -0 "$p" 2>/dev/null; then
                wait "$p" 2>/dev/null || true
                echo "[$(date '+%H:%M:%S')] pid=$p done (freed ${PIDG[$p]})"
                (( use_locks )) && _gpu_release ${PIDG[$p]//,/ }
                unset "PIDG[$p]"
            fi
        done

        # 2) Candidate pool = driver-free MINUS GPUs our own in-flight jobs hold.
        reserved=()
        for p in "${!PIDG[@]}"; do local g; for g in ${PIDG[$p]//,/ }; do reserved[$g]=1; done; done
        local -a avail=(); local idx
        for idx in $(gpu_nvidia_free "$thresh"); do [[ -z "${reserved[$idx]:-}" ]] && avail+=("$idx"); done

        # 3) Dispatch pending jobs (largest-first). Flock each GPU before taking
        #    it; a GPU another lane holds fails to lock and is left for later.
        local -a still=(); local placed=0 job
        for job in "${pending[@]}"; do
            local n="${job%%|*}" payload="${job#*|}"
            if (( n > total )); then
                echo "  WARN: job needs $n GPUs > $total available; clamping to $total" >&2
                n=$total
            fi
            local -a taken=() rest=()
            for idx in "${avail[@]}"; do
                if (( ${#taken[@]} < n )); then
                    if (( use_locks )); then
                        if _gpu_reserve "$idx"; then taken+=("$idx"); else rest+=("$idx"); fi
                    else
                        taken+=("$idx")
                    fi
                else
                    rest+=("$idx")
                fi
            done
            if (( ${#taken[@]} == n )); then
                local gpus; IFS=',' gpus="${taken[*]}"; unset IFS
                # Close inherited lock fds in the child so the job process never
                # pins another job's reservation; the lane keeps its own copies.
                ( if (( ${#_GPU_FD[@]} )); then for _fd in "${_GPU_FD[@]}"; do eval "exec ${_fd}>&-"; done 2>/dev/null; fi
                  "$launch_fn" "$gpus" "$payload" ) &
                PIDG[$!]="$gpus"; placed=1
                echo "[$(date '+%H:%M:%S')] dispatch pid=$! gpus=$gpus (still free: ${rest[*]:-none})"
            else
                (( use_locks )) && _gpu_release "${taken[@]}"
                still+=("$job")
            fi
            avail=("${rest[@]}")
        done
        pending=("${still[@]}")

        # 4) Block until state changes: sleep-poll while jobs are queued but
        #    unplaceable (to catch GPUs an external process frees); otherwise
        #    just wait for a running job to exit.
        if (( ${#pending[@]} > 0 && placed == 0 )); then
            sleep "$poll"
        elif (( ${#pending[@]} == 0 && ${#PIDG[@]} > 0 )); then
            local fp=; wait -n -p fp 2>/dev/null || true
            if [[ -n "${fp:-}" && -n "${PIDG[$fp]+x}" ]]; then
                echo "[$(date '+%H:%M:%S')] pid=$fp done (freed ${PIDG[$fp]})"
                (( use_locks )) && _gpu_release ${PIDG[$fp]//,/ }
                unset "PIDG[$fp]"
            elif [[ -z "${fp:-}" ]]; then
                (( use_locks )) && for p in "${!PIDG[@]}"; do _gpu_release ${PIDG[$p]//,/ }; done
                PIDG=()
            fi
        fi
    done
}
