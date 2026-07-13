cmake_minimum_required(VERSION 3.24)

# Toolchain for the DGX Spark (GB10): clang++ 18 as host compiler, nvcc (CUDA 13)
# as the device compiler, targeting sm_121. clang cannot emit sm_121 device code
# directly (only knows up to CUDA 12.3), so device code goes through nvcc -ccbin
# clang++. Override any of these with -D on the command line.
if(NOT DEFINED CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER clang)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER clang++)
endif()
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER clang++)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 121a)   # 'a' = arch-specific (enables Blackwell FP4 cvt on GB10)
endif()

project(lasybitstream LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Loader + device round-trip (the checkpoint-load milestone).
add_executable(lbload
  src/main.cc
  src/loader.cc
  cuda/device.cu)

# NVFP4 kernel correctness tests (match the compressed_tensors / CPU references).
add_executable(lbtest_nvfp4
  test/test_nvfp4.cc
  cuda/nvfp4.cu)
target_link_libraries(lbtest_nvfp4 PRIVATE CUDA::cudart)

add_executable(lbtest_gemm
  test/test_gemm.cc
  cuda/nvfp4.cu)
target_link_libraries(lbtest_gemm PRIVATE CUDA::cudart)

add_executable(lbtest_ops
  test/test_ops.cc
  cuda/ops.cu)
target_link_libraries(lbtest_ops PRIVATE CUDA::cudart)

# Composed MLP block: down(silu(gate(x)) * up(x)) from the NVFP4 GEMM + SwiGLU.
add_executable(lbtest_mlp
  test/test_mlp.cc
  cuda/nvfp4.cu
  cuda/ops.cu)
target_link_libraries(lbtest_mlp PRIVATE CUDA::cudart)

# GDN gated-delta-net recurrence (the 48 linear-attention layers).
add_executable(lbtest_gdn
  test/test_gdn.cc
  cuda/gdn.cu)
target_link_libraries(lbtest_gdn PRIVATE CUDA::cudart)

# The native engine (checkpoint load + full forward + vision tower), shared by CLI + server.
set(ENGINE_SRC src/engine.cc src/vision_tower.cc
    cuda/nvfp4.cu cuda/ops.cu cuda/gdn.cu cuda/forward_ops.cu cuda/vision.cu)

# Full native forward pass: validate vs the golden dump, or generate.
add_executable(lbinfer src/infer.cc ${ENGINE_SRC})
target_link_libraries(lbinfer PRIVATE CUDA::cudart)

# Byte-level BPE tokenizer + chat template (CPU only).
add_executable(lbtest_tok
  test/test_tokenizer.cc
  src/tokenizer.cc)

# OpenAI-compatible HTTP server: tokenizer + chat template + native engine.
add_executable(lbserve src/server.cc src/tokenizer.cc ${ENGINE_SRC})
target_link_libraries(lbserve PRIVATE CUDA::cudart)

# Vision tower (model.visual.* ViT): verify merged image embeddings vs the golden dump.
add_executable(lbtest_vis test/test_vision.cc cuda/vision.cu cuda/forward_ops.cu)
target_link_libraries(lbtest_vis PRIVATE CUDA::cudart)

# NVMe + RAM hybrid hot-KV cache: verify spill-to-disk + reload integrity.
add_executable(lbtest_kv test/test_kvcache.cc src/kvcache.cc)
target_link_libraries(lbtest_kv PRIVATE CUDA::cudart)

set_target_properties(lbload lbtest_nvfp4 lbtest_gemm lbtest_ops lbtest_mlp lbtest_gdn lbinfer lbserve lbtest_vis lbtest_kv PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

foreach(t lbload lbtest_nvfp4 lbtest_gemm lbtest_ops lbtest_mlp lbtest_gdn lbinfer lbtest_tok lbserve lbtest_vis lbtest_kv)
  target_compile_options(${t} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 -lineinfo>)
endforeach()
