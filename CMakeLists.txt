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

# https://gitlab.kitware.com/cmake/cmake/-/issues/24562
# https://discourse.cmake.org/t/cmake-hip-compiler-rocm-root-is-not-set-correctly/13102
# https://github.com/pytorch/pytorch/issues/128313#issuecomment-2167404020

list(APPEND CMAKE_PREFIX_PATH "${ROCM_SDK}")

message(STATUS "ROCm SDK: ${ROCM_SDK}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/lib64/cmake/hip/)
set(CMAKE_HIP_COMPILER_ROCM_ROOT ${ROCM_SDK})

project(
    fabsu
    VERSION 1.0
    DESCRIPTION "Fabsu"
    LANGUAGES CXX)

enable_language(${GPU_LANGUAGE})

set(CMAKE_${GPU_LANGUAGE}_STANDARD 20)
set(CMAKE_${GPU_LANGUAGE}_STANDARD_REQUIRED ON)
set(CMAKE_${GPU_LANGUAGE}_EXTENSIONS OFF)

set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem ${ROCM_SDK}/include")

message(STATUS "CMAKE_HIP_FLAGS: ${CMAKE_HIP_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

#set(CMAKE_${GPU_LANGUAGE}_FLAGS "${CMAKE_${GPU_LANGUAGE}_FLAGS} -isystem ${ROCM_SDK}/include")
#message(STATUS "CMAKE_<LANG>_FLAGS: ${CMAKE_${GPU_LANGUAGE}_FLAGS}")
