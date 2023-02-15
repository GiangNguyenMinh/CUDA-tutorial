cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(cuda_tutorial CUDA)

set(CMAKE_CUDA_STANDARD 17)

# CUDA
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

#
add_subdirectory(Part1_ProgramingModel)
add_subdirectory(Part2_ComplierNvcc)
add_subdirectory(Part3_CUDARuntime)


add_executable(cuda_tutorial main.cu)
target_link_libraries(cuda_tutorial cudart)

set_target_properties(cuda_tutorial PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
