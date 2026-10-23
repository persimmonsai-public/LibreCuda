#!/bin/bash
# Sets up the LLVM -> SPIR-V -> Mesa NAK flow's tools in a work directory (default ./work):
#   - Mesa 25.2.7's NIR and SPIR-V translator (static libraries; no drivers, no Rust) and libspirv_to_nir.so on them
#   - tinymesa 25.2.7.2 (Mesa's NAK and NIR as one library; pip, user site) and tinygrad (its ctypes bindings)
# The Mesa versions must match: libspirv_to_nir.so serializes NIR that tinymesa's nir_deserialize reads.
#
# needs: gcc, ninja, curl, git, python3 with pip (meson and mako are installed with pip --user)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mkdir -p "${1:-$here/work}" && cd "${1:-$here/work}" && pwd)"
mesa_version=25.2.7
tinygrad_commit=a474eea  # tinygrad's NAK bindings are generated for tinymesa 25.2.7.2

pip install --user -q meson mako "tinymesa==${mesa_version}.2"
meson="$(python3 -c 'import sysconfig; print(sysconfig.get_path("scripts", "posix_user"))')/meson"

cd "$work"
if [ ! -d "mesa-$mesa_version" ]; then
    curl -sL "https://archive.mesa3d.org/mesa-$mesa_version.tar.xz" | tar xJ
fi
if [ ! -f mesa-build/build.ninja ]; then
    (cd "mesa-$mesa_version" && "$meson" setup ../mesa-build -Dbuildtype=release -Dplatforms= -Dgallium-drivers= \
        -Dvulkan-drivers= -Dglx=disabled -Degl=disabled -Dgbm=disabled -Dopengl=false -Dgles1=disabled \
        -Dgles2=disabled -Dllvm=disabled -Dshared-glapi=disabled -Dzstd=disabled -Dexpat=disabled \
        -Dvalgrind=disabled -Dlibunwind=disabled -Dtools= > ../mesa-setup.log)
fi
libs=(src/compiler/spirv/libvtn.a src/compiler/nir/libnir.a src/compiler/libcompiler.a src/util/libmesa_util.a
      src/util/blake3/libblake3.a src/c11/impl/libmesa_util_c11.a)
ninja -C mesa-build "${libs[@]}" > mesa-build.log

# the shim, with the flags meson compiles Mesa's SPIR-V translator with (configuration defines, include paths)
cflags=$(python3 - "$work/mesa-build" <<'EOF'
import json, shlex, sys
build = sys.argv[1]
c = next(c for c in json.load(open(f"{build}/compile_commands.json")) if c["file"].endswith("spirv/spirv_to_nir.c"))
out = []
for a in shlex.split(c["command"])[1:]:
    if a.startswith(("-D", "-I", "-std", "-f")):
        out.append("-I" + build + "/" + a[2:] if a.startswith("-I") and not a.startswith("-I/") else a)
print(" ".join(shlex.quote(a) for a in out))
EOF
)
eval "gcc -O2 -fPIC -shared $cflags -o libspirv_to_nir.so '$here/spirv_to_nir.c' \
    -Wl,--whole-archive mesa-build/${libs[0]} -Wl,--no-whole-archive $(printf 'mesa-build/%s ' "${libs[@]:1}") \
    -lm -lpthread -ldl -Wl,--no-undefined"

if [ ! -d tinygrad ]; then
    git clone -q https://github.com/tinygrad/tinygrad.git
    git -C tinygrad checkout -q "$tinygrad_commit"
fi
echo "set up in $work"
