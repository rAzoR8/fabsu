cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(GPU_LANGUAGE "HIP" CACHE STRING "GPU Lang: HIP or CUDA")
set(PROJECT_LANGUAGE "CXX" CACHE STRING "Project Lang: HIP or CXX")

if(WIN32)
    set(ROCM_SDK "$ENV(HIP_PATH)" CACHE PATH "HIP / ROCm SDK Dir") 
else()
    #set(ROCM_SDK "/opt/rocm" CACHE PATH "HIP / ROCm SDK Dir")

    # not all distors install rocm to /opt/rocm!
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
# https://github.com/ROCm/HIP/issues/3406
# https://lists.llvm.org/pipermail/cfe-commits/Week-of-Mon-20250106/660706.html
# https://cgmb-rocm-docs.readthedocs.io/en/latest/Installation_Guide/Using-CMake-with-AMD-ROCm.html
# https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html
# https://github.com/pelagos-consulting/HIP_Course/blob/main/course_material/L4_Debugging/CMakeLists.txt

set(CMAKE_HIP_COMPILER_ROCM_ROOT ${ROCM_SDK})
set(CMAKE_${GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${GPU_LANGUAGE}_STANDARD_REQUIRED ON)
set(CMAKE_${GPU_LANGUAGE}_EXTENSIONS OFF)

project(
    fabsu
    VERSION 1.0
    DESCRIPTION "Fabsu"
    LANGUAGES ${PROJECT_LANGUAGE}) # when using ${GPU_LANGUAGE} or HIP here, VS Code syntax highlighting does not work

message(STATUS "ROCm SDK: ${ROCM_SDK}")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_SDK}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

enable_language(${GPU_LANGUAGE})

# due to non standard behaviour of hip clang, we need to completely disable any optimization to be able to debug
set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG} -O0") 

message(STATUS "PROJECT_LANGUAGE: ${PROJECT_LANGUAGE}")
message(STATUS "GPU_LANGUAGE: ${GPU_LANGUAGE}")
message(STATUS "CMAKE_HIP_FLAGS_DEBUG: ${CMAKE_HIP_FLAGS_DEBUG}")
message(STATUS "CMAKE_HIP_FLAGS: ${CMAKE_HIP_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_HIP_ARCHITECTURES:" ${CMAKE_HIP_ARCHITECTURES})

# set(fabsu_app_INC
#     ${CMAKE_CURRENT_SOURCE_DIR}/include/fabsu.hpp
# )

set(fabsu_app_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fabsu.cpp
)

add_executable(fabsu_app ${fabsu_app_SRC})

if(PROJECT_LANGUAGE STREQUAL "CXX")
    find_package(hip REQUIRED) # needed when project lang is CXX
    target_link_libraries(fabsu_app PRIVATE hip::host)
    target_link_libraries(fabsu_app PRIVATE hip::device)
endif()

set_target_properties(fabsu_app PROPERTIES LINKER_LANGUAGE ${GPU_LANGUAGE})
set_source_files_properties(${fabsu_app_SRC} PROPERTIES LANGUAGE ${GPU_LANGUAGE})

target_include_directories(fabsu_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(fabsu_app PRIVATE ${ROCM_SDK}/include) # this is not enough to make VSCode find <hip/hip_runtime.h>