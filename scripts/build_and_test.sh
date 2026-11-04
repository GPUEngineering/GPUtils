#!/usr/bin/env bash
# Configure/build GPUtils and run every C++ and Python unit test.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${GPUTILS_BUILD_DIR:-$repo_root/build}"
case "$build_dir" in /*) ;; *) build_dir="$repo_root/$build_dir" ;; esac
python="${PYTHON:-python3}"
if [[ -z "${PYTHON:-}" && -x "$build_dir/test-venv/bin/python" ]]; then
    python="$build_dir/test-venv/bin/python"
fi
jobs="${JOBS:-4}"

if ! "$python" -c 'import numpy' >/dev/null 2>&1; then
    echo "Python with NumPy is required. Install python/ in a virtual environment and set PYTHON to its interpreter." >&2
    exit 1
fi

# Extra arguments are forwarded to CMake, e.g. -DSM_ARCH=120 -DCPPVERSION=20.
cmake -S "$repo_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo "$@" -DGPUTILS_BUILD_TEST=ON
cmake --build "$build_dir" --parallel "$jobs"
ctest --test-dir "$build_dir/test" --output-on-failure --no-tests=error

# The Python tests write fixtures to their working directory. Keep those in
# the build tree rather than the source checkout.
mkdir -p "$build_dir/python-tests"
(
    cd -- "$build_dir/python-tests"
    PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}" \
        "$python" "$repo_root/python/test/test.py" -v
)
