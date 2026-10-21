#!/bin/bash
# Compiles the CUDA runtime test programs with nvcc and runs them on the LibreCUDA runtime.
#
# usage: run_tests.sh [build dir] [test names...]
#   build dir defaults to ../../build; the LibreCUDA runtime is expected in <build dir>/lib.
#   Set NVCC to choose the compiler, ARCH to choose the target (default sm_120). Tests are only rebuilt when their
#   source, check.h or the build command changed; FORCE_REBUILD=1 rebuilds all. A test made of several translation
#   units lists the others in a "// EXTRA_SOURCES:" line (kept in parts/, which isn't searched for tests).
set -u
script_dir="$(cd "$(dirname "$0")" && pwd)"
build_dir="$(cd "${1:-$script_dir/../../build}" && pwd)"
shift || true
nvcc="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
arch="${ARCH:-sm_120}"
lib_dir="$build_dir/lib"
out_dir="$build_dir/tests/cudart"
mkdir -p "$out_dir"

if [ ! -f "$lib_dir/libcudart.so.13" ]; then
    echo "LibreCUDA runtime not found in $lib_dir" >&2
    exit 1
fi

tests=("$@")
if [ ${#tests[@]} -eq 0 ]; then
    for f in "$script_dir"/*.cu; do tests+=("$(basename "$f" .cu)"); done
fi

failed=()
for t in "${tests[@]}"; do
    src="$script_dir/$t.cu"
    exe="$out_dir/$t"
    extra_flags=""
    # tests may request extra nvcc flags with a "// NVCC_FLAGS: ..." line
    flags_line=$(grep -m1 '^// NVCC_FLAGS:' "$src" | sed 's|^// NVCC_FLAGS:||')
    [ -n "$flags_line" ] && extra_flags="${flags_line//\$ARCH/$arch}" || extra_flags="-arch=$arch"
    # further translation units of the program: "// EXTRA_SOURCES: parts/a.cu parts/b.cu" (relative to this directory)
    extra_sources=()
    for extra in $(grep -m1 '^// EXTRA_SOURCES:' "$src" | sed 's|^// EXTRA_SOURCES:||'); do
        extra_sources+=("$script_dir/$extra")
    done
    sources_changed=0
    for extra in "${extra_sources[@]}"; do
        [ "$extra" -nt "$exe" ] && sources_changed=1
    done
    # -cudart=shared links against libcudart.so.13, which resolves to the LibreCUDA runtime at run time, so a rebuilt
    # runtime needs no relink. Rebuild only when the source, check.h or the flags changed (FORCE_REBUILD=1 always does).
    build_cmd="$nvcc -O3 -std=c++17 $extra_flags -cudart=shared -I$script_dir $src ${extra_sources[*]} -o $exe"
    if [ "${FORCE_REBUILD:-0}" != 0 ] || [ ! -x "$exe" ] || [ "$src" -nt "$exe" ] || [ "$script_dir/check.h" -nt "$exe" ] ||
       [ "$sources_changed" != 0 ] || [ "$(cat "$exe.cmd" 2>/dev/null)" != "$build_cmd" ]; then
        if ! $build_cmd 2> "$exe.build.log"; then
            echo "BUILD FAILED $t (see $exe.build.log)"
            rm -f "$exe" "$exe.cmd"
            failed+=("$t")
            continue
        fi
        echo "$build_cmd" > "$exe.cmd"
    fi
    start=$(date +%s.%N)
    # a GPU side hang turns into an error instead of an endless wait
    if LIBRECUDA_WAIT_TIMEOUT_MS="${LIBRECUDA_WAIT_TIMEOUT_MS:-60000}" LD_LIBRARY_PATH="$lib_dir" timeout 600 "$exe" > "$exe.log" 2>&1; then
        end=$(date +%s.%N)
        printf "%-28s ok   (%.2fs)\n" "$t" "$(echo "$end - $start" | bc)"
        grep -v '^PASS' "$exe.log" | sed 's/^/    /'
    else
        echo "$t FAILED (exit $?)"
        sed 's/^/    /' "$exe.log" | tail -20
        failed+=("$t")
    fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "all ${#tests[@]} tests passed"
else
    echo "${#failed[@]} of ${#tests[@]} tests failed: ${failed[*]}"
    exit 1
fi
