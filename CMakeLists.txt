cmake_minimum_required(VERSION 3.17)
project (RTX-RMQ LANGUAGES CXX CUDA)

# Set default build type to 'Release'
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# add path to our CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

# turn on C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# for common headers
include_directories(./)
include_directories(./include)

# ==============================================================================
# Utility function to get PTX compilation & copying working
# ==============================================================================
include(PTXUtilities)

# ==============================================================================
# To link against CUDA::cudart and/or CUDA::cuda_driver
# ==============================================================================
find_package(CUDAToolkit REQUIRED)
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-O3 -fPIE")

# ==============================================================================
# Find Optix7
# ==============================================================================
find_package(OptiX7 REQUIRED)
include_directories(${OptiX7_INCLUDE_DIRS})


# ==============================================================================
# Find OpenMP Library
# ==============================================================================
find_package(OpenMP REQUIRED)
IF (OPENMP_FOUND)
    # SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-O3,${OpenMP_CXX_FLAGS}")
    # SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
 
ENDIF()

# ==============================================================================
# Generate compile-time constants header
# ==============================================================================

# Provide defaults if not passed from command line
if(NOT DEFINED BSIZE)
    set(BSIZE 1024)
endif()

if(NOT DEFINED IS_LONG)
    set(IS_LONG 0)
endif()

if(NOT DEFINED XXX_CG_SIZE_LOG)
    set(XXX_CG_SIZE_LOG 3)
endif()

if(NOT DEFINED XXX_CG_AMOUNT_LOG)
    set(XXX_CG_AMOUNT_LOG 2)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/constants_config.cuh.in
    ${CMAKE_CURRENT_BINARY_DIR}/constants_config.cuh
    @ONLY
)

# ==============================================================================
# Main Executable
# ==============================================================================
find_library(RMQRMM rmqrmmBP.a hrmq/)

add_executable(rtxrmq src/main.cu common/common.cpp common/Rapl.cpp common/nvmlPower.cpp src/xxx.h)
target_include_directories(rtxrmq PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(rtxrmq PRIVATE CUDA::cudart CUDA::nvml OptiX7 OpenMP::OpenMP_CXX ${RMQRMM})
set_target_properties(rtxrmq PROPERTIES CUDA_ARCHITECTURES "86;89;120")
IF (OPENMP_FOUND)
    target_link_libraries(rtxrmq PUBLIC OpenMP::OpenMP_CXX)
endif()

# ==============================================================================
# Add any needed PTX compilations here
# ==============================================================================
add_ptx_targets(rtxrmq rtx_kernels)
target_include_directories(rtxrmq_rtx_kernels PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
