/*
 * SPIR-V (OpenCL environment, as LLVM's SPIR-V backend emits it) -> NIR for NAK, with LibreCUDA's kernel ABI, as
 * serialized NIR. Built against Mesa's NIR and SPIR-V translator (the same Mesa version as the NAK library that
 * deserializes it: tinymesa 25.2.7 exports nir_deserialize and NAK but not spirv_to_nir).
 *
 * The lowering follows rusticl's (src/gallium/frontends/rusticl/core/kernel.rs) minus libclc, printf, images and
 * the internal kernel arguments; kernel parameters, the block and the grid size then become loads from constant
 * bank 0 at LibreCUDA's offsets:
 *   c[0][0x360 + 4i] blockDim, c[0][0x370 + 4i] gridDim, c[0][param_base + offset] parameters (sm_90+: 0x380)
 */
#include "compiler/glsl_types.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv_info.h"
#include "util/blob.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_DIM_OFFSET 0x360
#define GRID_DIM_OFFSET 0x370

struct abi {
    unsigned param_base;
};

static nir_def *ldc(nir_builder *b, unsigned components, unsigned bits, nir_def *offset) {
    return nir_ldc_nv(b, components, bits, nir_imm_int(b, 0), offset, .align_mul = bits / 8, .align_offset = 0);
}

static bool lower_abi(nir_builder *b, nir_intrinsic_instr *intrin, void *data) {
    const struct abi *abi = data;
    b->cursor = nir_before_instr(&intrin->instr);
    nir_def *value;
    switch (intrin->intrinsic) {
        case nir_intrinsic_load_kernel_input:
            value = ldc(b, intrin->def.num_components, intrin->def.bit_size,
                        nir_iadd_imm(b, intrin->src[0].ssa, abi->param_base + nir_intrinsic_base(intrin)));
            break;
        case nir_intrinsic_load_workgroup_size:
            value = ldc(b, 3, 32, nir_imm_int(b, BLOCK_DIM_OFFSET));
            break;
        case nir_intrinsic_load_num_workgroups:
            value = ldc(b, 3, 32, nir_imm_int(b, GRID_DIM_OFFSET));
            break;
        default:
            return false;
    }
    nir_def_replace(&intrin->def, nir_u2uN(b, value, intrin->def.bit_size));
    return true;
}

static void log_spirv(void *data, enum nir_spirv_debug_level level, size_t offset, const char *message) {
    if (level >= NIR_SPIRV_DEBUG_LEVEL_WARNING) fprintf(stderr, "spirv_to_nir (word %zu): %s\n", offset / 4, message);
}

static void optimize(nir_shader *nir) {
    bool progress;
    do {
        progress = false;
        NIR_PASS(_, nir, nir_split_var_copies);
        NIR_PASS(progress, nir, nir_copy_prop);
        NIR_PASS(progress, nir, nir_opt_copy_prop_vars);
        NIR_PASS(progress, nir, nir_opt_dead_write_vars);
        NIR_PASS(progress, nir, nir_opt_deref);
        NIR_PASS(progress, nir, nir_opt_dce);
        NIR_PASS(progress, nir, nir_opt_undef);
        NIR_PASS(progress, nir, nir_opt_constant_folding);
        NIR_PASS(progress, nir, nir_opt_cse);
        NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
        NIR_PASS(progress, nir, nir_opt_algebraic);
        NIR_PASS(progress, nir, nir_opt_dead_cf);
        NIR_PASS(progress, nir, nir_opt_remove_phis);
    } while (progress);
}

/*
 * Translates kernel `entry` of the SPIR-V module. On success returns 0, the serialized NIR in *out (free()), its
 * static shared memory size and the parameters' offsets / sizes (in parameter order, at most max_params).
 */
__attribute__((visibility("default"))) int lc_spirv_to_nir(const uint32_t *words, size_t word_count, const char *entry, const void *nak_nir_options,
                    unsigned param_base, void **out, size_t *out_size, unsigned *shared_size, unsigned *num_params,
                    unsigned *param_offsets, unsigned *param_sizes, unsigned max_params) {
    glsl_type_singleton_init_or_ref();
    const struct spirv_capabilities caps = {
        .Addresses = true, .Linkage = true, .Kernel = true, .Float16 = true, .Float64 = true, .Int64 = true,
        .Int64Atomics = true, .Int8 = true, .Int16 = true, .Groups = true, .GenericPointer = true,
    };
    const struct spirv_to_nir_options spirv_options = {
        .environment = NIR_SPIRV_OPENCL,
        .capabilities = &caps,
        .shared_addr_format = nir_address_format_32bit_offset_as_64bit,
        .global_addr_format = nir_address_format_64bit_global,
        .temp_addr_format = nir_address_format_32bit_offset_as_64bit,
        .constant_addr_format = nir_address_format_64bit_global,
        .debug = {.func = log_spirv},
    };
    nir_shader *nir = spirv_to_nir(words, word_count, NULL, 0, MESA_SHADER_KERNEL, entry, &spirv_options,
                                   (const nir_shader_compiler_options *) nak_nir_options);
    if (nir == NULL) return 1;
    nir->info.workgroup_size_variable = true;  // the block size comes from the launch

    NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
    NIR_PASS(_, nir, nir_lower_returns);
    NIR_PASS(_, nir, nir_inline_functions);
    nir_remove_non_entrypoints(nir);
    optimize(nir);
    NIR_PASS(_, nir, nir_lower_variable_initializers, ~nir_var_function_temp);
    NIR_PASS(_, nir, nir_lower_system_values);
    const nir_lower_compute_system_values_options compute_options = {0};
    NIR_PASS(_, nir, nir_lower_compute_system_values, &compute_options);

    // (the module's other kernels' shared variables, before they get offsets)
    NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_mem_shared | nir_var_function_temp, NULL);
    NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
             nir_var_mem_shared | nir_var_function_temp | nir_var_shader_temp | nir_var_uniform | nir_var_mem_global |
             nir_var_mem_generic, glsl_get_cl_type_size_align);
    // the parameters, in order (their offsets were assigned by the explicit types)
    unsigned n = 0;
    nir_foreach_variable_with_modes(var, nir, nir_var_uniform) {
        const unsigned index = var->data.location;
        if (index >= max_params) return 2;
        unsigned size, align;
        glsl_get_cl_type_size_align(var->type, &size, &align);
        param_offsets[index] = var->data.driver_location;
        param_sizes[index] = size;
        if (index + 1 > n) n = index + 1;
    }
    *num_params = n;

    optimize(nir);
    NIR_PASS(_, nir, nir_lower_memcpy);
    NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global | nir_var_mem_constant,
             nir_address_format_64bit_global);
    NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared | nir_var_function_temp | nir_var_uniform,
             nir_address_format_32bit_offset_as_64bit);
    NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_generic, nir_address_format_62bit_generic);
    struct abi abi = {param_base};
    NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_abi, nir_metadata_control_flow, &abi);
    NIR_PASS(_, nir, nir_lower_convert_alu_types, NULL);
    optimize(nir);
    NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_uniform | nir_var_mem_shared | nir_var_function_temp, NULL);
    // NAK (25.2) compiles compute shaders, not kernels; nothing kernel specific is left
    nir->info.stage = MESA_SHADER_COMPUTE;
    // NAK limits the registers by the block size: assume the largest one (1024 threads, at most 64 registers)
    nir->info.workgroup_size[0] = 1024;
    nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
    nir_validate_shader(nir, "lc_spirv_to_nir");

    if (getenv("LC_PRINT_NIR") != NULL) nir_print_shader(nir, stderr);
    *shared_size = nir->info.shared_size;
    struct blob blob;
    blob_init(&blob);
    nir_serialize(&blob, nir, false);
    *out_size = blob.size;
    *out = malloc(blob.size);
    memcpy(*out, blob.data, blob.size);
    blob_finish(&blob);
    ralloc_free(nir);
    return 0;
}
