#!/usr/bin/env bash
# Run all Python tests with pytest; forward any extra pytest arguments.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${GPUTILS_BUILD_DIR:-$repo_root/build}"
case "$build_dir" in /*) ;; *) build_dir="$repo_root/$build_dir" ;; esac
python="${PYTHON:-python3}"
if [[ -z "${PYTHON:-}" && -x "$build_dir/test-venv/bin/python" ]]; then
    python="$build_dir/test-venv/bin/python"
fi

if ! "$python" -c 'import numpy, pytest' >/dev/null 2>&1; then
    echo "Python with NumPy and pytest is required. Install them in a virtual environment and set PYTHON to its interpreter." >&2
    exit 1
fi

# Reuse a built helper for the C++ exchange tests; honour an explicit override.
# The Python suite reports those tests as skipped if no helper is available.
if [[ -z "${GPUTILS_FILE_EXCHANGE:-}" && -x "$build_dir/test/gputils_file_exchange" ]]; then
    export GPUTILS_FILE_EXCHANGE="$build_dir/test/gputils_file_exchange"
fi
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m pytest "$repo_root/python/test" -v "$@"
