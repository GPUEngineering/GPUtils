#!/usr/bin/env bash
# Build/test, then check the full CUDA suite and main executable for leaks.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${GPUTILS_BUILD_DIR:-$repo_root/build}"
case "$build_dir" in /*) ;; *) build_dir="$repo_root/$build_dir" ;; esac
sanitizer="${COMPUTE_SANITIZER:-/usr/local/cuda/bin/compute-sanitizer}"
repeats="${SANITIZER_REPEATS:-3}"
if ! command -v "$sanitizer" >/dev/null 2>&1; then
    echo "Compute Sanitizer not found; set COMPUTE_SANITIZER to its path." >&2
    exit 1
fi
if [[ ! "$repeats" =~ ^[1-9][0-9]*$ ]]; then
    echo "SANITIZER_REPEATS must be a positive integer." >&2
    exit 1
fi

bash "$repo_root/scripts/build_and_test.sh" \
    '-DCMAKE_CUDA_FLAGS_RELWITHDEBINFO=-O2 -g -lineinfo -DNDEBUG' "$@"
mkdir -p "$build_dir/sanitizer"
cd -- "$build_dir/sanitizer"
options=(--tool memcheck --leak-check full --check-device-heap yes --error-exitcode 1)

# One process, shuffled repetitions: catches leaks and stale ownership retained
# across tests/repeated use as well as invalid device accesses. pipefail preserves
# sanitizer or test failures while tee keeps complete logs.
"$sanitizer" "${options[@]}" "$build_dir/test/gputils_test" \
    --gtest_shuffle --gtest_random_seed=1 --gtest_repeat="$repeats" \
    2>&1 | tee tests-memcheck.log
"$sanitizer" "${options[@]}" "$build_dir/gputils_main" \
    2>&1 | tee main-memcheck.log

# Fail closed if instrumentation did not produce a successful completion report.
for report in tests-memcheck.log main-memcheck.log; do
    if ! grep -Eq 'ERROR SUMMARY: 0 errors' "$report"; then
        echo "Missing clean sanitizer summary: $report" >&2
        exit 1
    fi
    if ! grep -Eq 'LEAK SUMMARY: 0 bytes leaked in 0 allocations' "$report"; then
        echo "Leak check failed: $report" >&2
        exit 1
    fi
done
echo "CUDA memory checks passed; logs: $build_dir/sanitizer"
