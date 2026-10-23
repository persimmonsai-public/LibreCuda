#!/usr/bin/env python3
"""
SPIR-V -> SM_120 cubins with Mesa's NAK: every kernel (OpEntryPoint) of the module is translated to NIR by
libspirv_to_nir.so (spirv_to_nir.c: Mesa's SPIR-V translator plus LibreCUDA's kernel ABI), compiled by NAK
(tinymesa, through tinygrad's bindings) and written as <out dir>/<kernel>.cubin, a minimal CUDA ELF that LibreCUDA's
cuModuleLoadData loads (the format of tools/nak_cubin.py, plus .nv.shared for static shared memory).

usage: nak_compile.py <module.spv> <out dir> <libspirv_to_nir.so> [--sm 120]
needs: tinygrad on PYTHONPATH, MESA_PATH = tinymesa 25.2.7's libtinymesa.so (the version libspirv_to_nir.so was built
from: serialized NIR is version specific)
"""
import argparse
import base64
import ctypes
import struct
import sys
from pathlib import Path

from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.support.compiler_mesa import NAKCompiler

PARAM_BASE = {120: 0x380}  # Blackwell: parameters follow 224 dwords of driver data in constant bank 0
MAX_PARAMS = 64


def entry_points(words: list[int]) -> list[str]:
    names, i = [], 5
    while i < len(words):
        count, opcode = words[i] >> 16, words[i] & 0xffff
        if opcode == 15:  # OpEntryPoint: execution model, id, name...
            raw = struct.pack(f"<{count - 3}I", *words[i + 3:i + count])
            names.append(raw[:raw.index(b"\0")].decode())
        i += count
    return names


def compile_kernel(shim, cc: NAKCompiler, words, name: str, sm: int) -> dict:
    out, out_size = ctypes.c_void_p(), ctypes.c_size_t()
    shared, nparams = ctypes.c_uint(), ctypes.c_uint()
    offsets, sizes = (ctypes.c_uint * MAX_PARAMS)(), (ctypes.c_uint * MAX_PARAMS)()
    options = mesa.nak_nir_options(cc.cc)
    if (rc := shim.lc_spirv_to_nir(words, len(words), name.encode(), ctypes.cast(options, ctypes.c_void_p),
                            PARAM_BASE.get(sm, 0x160), ctypes.byref(out), ctypes.byref(out_size),
                            ctypes.byref(shared), ctypes.byref(nparams), offsets, sizes, MAX_PARAMS)) != 0:
        raise RuntimeError(f"SPIR-V to NIR failed ({rc})")
    blob = ctypes.string_at(out, out_size.value)
    ctypes.CDLL(None).free(out)
    binary = cc.compile(base64.b64encode(blob).decode())
    n = ctypes.sizeof(mesa.struct_nak_shader_info)
    info = mesa.struct_nak_shader_info.from_buffer_copy(binary[:n])
    params = [(offsets[i], sizes[i]) for i in range(nparams.value)]
    param_size = max((o + s for o, s in params), default=0)
    return dict(name=name, code=binary[n:], param_base=PARAM_BASE.get(sm, 0x160), num_gprs=info.num_gprs, slm_size=info.slm_size,
                smem_size=max(shared.value, info.cs.smem_size), params=params, param_size=param_size,
                cbuf0_size=(PARAM_BASE.get(sm, 0x160) + param_size + 3) // 4 * 4)


# ---- minimal cubin ELF (what libreCuModuleLoadData parses)

SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB, SHT_NOBITS, SHT_NV_INFO = 1, 2, 3, 8, 0x70000000
SHF_WRITE, SHF_ALLOC, SHF_EXECINSTR = 0x1, 0x2, 0x4
EIATTR_KPARAM_INFO, EIATTR_CBANK_PARAM_SIZE, EIATTR_MIN_STACK_SIZE, EIATTR_PARAM_CBANK = 0x1704, 0x1903, 0x1204, 0x0a04


class StrTab:
    def __init__(self):
        self.data = bytearray(b"\0")

    def add(self, s: str) -> int:
        off = len(self.data)
        self.data += s.encode() + b"\0"
        return off


def write_cubin(k: dict, sm: int) -> bytes:
    shstr, strtab = StrTab(), StrTab()
    name = k["name"]
    info = bytearray()
    for i, (off, size) in enumerate(k["params"]):
        info += struct.pack("<IIII", EIATTR_KPARAM_INFO | (12 << 16), 0, i | (off << 16), 0x1f000 | (size << 18))
    info += struct.pack("<I", EIATTR_CBANK_PARAM_SIZE | (k["param_size"] << 16))
    # where the parameters start in constant bank 0: (size << 16) | offset (the bank's symbol isn't needed)
    info += struct.pack("<III", EIATTR_PARAM_CBANK | (8 << 16), 0, (k["param_size"] << 16) | k["param_base"])
    # (name, type, flags, data or size, info, align, entsize, link)
    sections = [None, [".shstrtab", SHT_STRTAB, 0, None, 0, 1, 0, 0], [".strtab", SHT_STRTAB, 0, None, 0, 1, 0, 0],
                [".symtab", SHT_SYMTAB, 0, None, 0, 8, 24, 2],
                [f".nv.info.{name}", SHT_NV_INFO, 0, bytes(info), 0, 4, 0, 3],
                [f".nv.constant0.{name}", SHT_PROGBITS, SHF_ALLOC, bytes(k["cbuf0_size"]), 0, 64, 0, 0]]
    text = len(sections)
    # code padded so instruction prefetch past the end stays inside the section
    sections.append([f".text.{name}", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, k["code"] + bytes(0x100),
                     (k["num_gprs"] & 0xff) << 24, 128, 0, 3])
    if k["smem_size"]:
        sections.append([f".nv.shared.{name}", SHT_NOBITS, SHF_WRITE | SHF_ALLOC, k["smem_size"], 0, 16, 0, 0])
    if k["slm_size"]:
        sections.append([".nv.info", SHT_NV_INFO, 0,
                         struct.pack("<III", EIATTR_MIN_STACK_SIZE | (8 << 16), 1, k["slm_size"]), 0, 4, 0, 3])
    syms = struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0)
    syms += struct.pack("<IBBHQQ", strtab.add(name), (1 << 4) | 2, 0, text, 0, len(k["code"]))  # GLOBAL FUNC
    sections[3][3], sections[3][4] = syms, 1
    for s in sections[1:]:
        s.insert(0, shstr.add(s[0]))
    sections[2][4] = bytes(strtab.data)
    sections[1][4] = bytes(shstr.data)

    body, offsets, sizes = bytearray(), [], []
    for s in sections[1:]:
        data, align = s[4], s[6]
        body += bytes((-(64 + len(body))) % align)
        offsets.append(64 + len(body))
        if s[2] == SHT_NOBITS:
            sizes.append(data)
        else:
            sizes.append(len(data))
            body += data
    shoff = 64 + len(body) + ((-(64 + len(body))) % 8)
    body += bytes(shoff - 64 - len(body))
    shdrs = bytes(64)
    for s, off, size in zip(sections[1:], offsets, sizes):
        shname, _, typ, flags, _, info_field, align, entsize, link = s
        shdrs += struct.pack("<IIQQQQIIQQ", shname, typ, flags, 0, off, size, link, info_field, align, entsize)
    # e_flags carries the SM version like nvcc's cubins; e_machine 190 = EM_CUDA
    ehdr = b"\x7fELF" + bytes([2, 1, 1, 0x33, 7]) + bytes(7)
    ehdr += struct.pack("<HHIQQQIHHHHHH", 2, 190, 1, 0, 0, shoff, sm | (sm << 16), 64, 0, 0, 64, len(sections), 1)
    return bytes(ehdr + body + shdrs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spv")
    ap.add_argument("out_dir")
    ap.add_argument("shim")
    ap.add_argument("--sm", type=int, default=120)
    args = ap.parse_args()
    data = Path(args.spv).read_bytes()
    words = list(struct.unpack(f"<{len(data) // 4}I", data))
    shim = ctypes.CDLL(args.shim)
    shim.lc_spirv_to_nir.restype = ctypes.c_int
    shim.lc_spirv_to_nir.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t, ctypes.c_char_p,
                                     ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_uint),
                                     ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
                                     ctypes.POINTER(ctypes.c_uint), ctypes.c_uint]
    spirv = (ctypes.c_uint32 * len(words))(*words)
    mesa.glsl_type_singleton_init_or_ref()
    cc = NAKCompiler(f"sm_{args.sm}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name in entry_points(words):
        try:
            k = compile_kernel(shim, cc, spirv, name, args.sm)
        except Exception as e:
            print(f"{name}: {e}", file=sys.stderr)
            failed += 1
            continue
        (out / f"{name}.cubin").write_bytes(write_cubin(k, args.sm))
        print(f"{name}: {len(k['code'])} bytes, {k['num_gprs']} registers, shared {k['smem_size']}, "
              f"local {k['slm_size']}, params {k['params']}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
