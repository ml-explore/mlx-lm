// Copyright © 2026 IndenScale

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "paged_sdpa.h"

using namespace metal;

#define instantiate_paged_sdpa(type, gqa)                               \
  instantiate_kernel(                                                   \
      "paged_sdpa_vector_2pass_1_gqa_" #gqa "_" #type "_256",        \
      paged_sdpa_vector_2pass_1_gqa,                                    \
      type,                                                             \
      256,                                                              \
      gqa,                                                              \
      2)

#define instantiate_paged_sdpa_types(type) \
  instantiate_paged_sdpa(type, 6)          \
  instantiate_paged_sdpa(type, 8)

instantiate_paged_sdpa_types(float)
instantiate_paged_sdpa_types(float16_t)
instantiate_paged_sdpa_types(bfloat16_t)

#include "mlx/backend/metal/kernels/sdpa_vector.h"
instantiate_kernel("eco_sdpa_reduce_float_256", sdpa_vector_2pass_2, float, 256)
instantiate_kernel("eco_sdpa_reduce_float16_t_256", sdpa_vector_2pass_2, float16_t, 256)
instantiate_kernel("eco_sdpa_reduce_bfloat16_t_256", sdpa_vector_2pass_2, bfloat16_t, 256)
