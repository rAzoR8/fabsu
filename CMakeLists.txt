cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(GPU_LANGUAGE "HIP" CACHE STRING "GPU Lang: HIP or CUDA")

if(WIN32)
    set(ROCM_SDK "$ENV(HIP_PATH)" CACHE PATH "HIP / ROCm SDK Dir") 
else()
    #set(ROCM_SDK "/opt/rocm" CACHE PATH "HIP / ROCm SDK Dir")

    find_program(HIP_CONFIG hipconfig HINTS "/opt/rocm" "/usr/bin")
    if(NOT HIP_CONFIG)
        message(FATAL_ERROR "hipconfig not found!")
    endif()

    execute_process(
        COMMAND ${HIP_CONFIG} --rocmpath OUTPUT_VARIABLE ROCM_SDK
    )
endif()

# https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html
# https://gitlab.kitware.com/cmake/cmake/-/issues/24562
# https://discourse.cmake.org/t/cmake-hip-compiler-rocm-root-is-not-set-correctly/13102
# https://github.com/pytorch/pytorch/issues/128313#issuecomment-2167404020

# list(APPEND CMAKE_PREFIX_PATH "${ROCM_SDK}")

message(STATUS "ROCm SDK: ${ROCM_SDK}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/lib64/cmake/hip/)
set(CMAKE_HIP_COMPILER_ROCM_ROOT ${ROCM_SDK})
set(CMAKE_${GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${GPU_LANGUAGE}_STANDARD_REQUIRED ON)
set(CMAKE_${GPU_LANGUAGE}_EXTENSIONS OFF)

#set(CMAKE_HIP_EXTENSIONS OFF)
#set(CMAKE_CXX_STANDARD 17) 
#set(CMAKE_CXX_STANDARD_REQUIRED ON) 

project(
    fabsu
    VERSION 1.0
    DESCRIPTION "Fabsu"
    LANGUAGES HIP)

enable_language(${GPU_LANGUAGE})

# TODO: this breaks stdlibc cmath.h include
# but without this VS Code intellisense does not work.
#set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem ${ROCM_SDK}/include")

message(STATUS "CMAKE_HIP_FLAGS: ${CMAKE_HIP_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_HIP_ARCHITECTURES:" ${CMAKE_HIP_ARCHITECTURES})

set(fabsu_INC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/fabsu.hpp
)

set(fabsu_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fabsu.hip
)

add_executable(fabsu ${fabsu_SRC})
target_include_directories(fabsu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(fabsu PRIVATE ${ROCM_SDK}/include)

# https://github.com/ROCm/HIP/issues/3406
# https://lists.llvm.org/pipermail/cfe-commits/Week-of-Mon-20250106/660706.html

#set_source_files_properties(${fabsu_SRC} PROPERTIES LANGUAGE ${GPU_LANGUAGE})