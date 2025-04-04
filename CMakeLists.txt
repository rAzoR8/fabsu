cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(GPU_LANGUAGE "HIP" CACHE STRING "GPU Lang: HIP or CUDA")
set(PROJECT_LANGUAGE "CXX" CACHE STRING "Project Lang: HIP or CXX")
set(ADVANCED_DEBUGPROF OFF CACHE BOOL "Dump code objects and emit profiling info")

# Use this option when using the docker Dev Container to get VSCode InstelliSense to find hip_runtime.h
set(ROCM_SYSTEM_INCLUDE OFF CACHE BOOL "Force include -isystem rocm-path/include (for Dev Conainer)")

message(STATUS "PROJECT_LANGUAGE: ${PROJECT_LANGUAGE}")
message(STATUS "GPU_LANGUAGE: ${GPU_LANGUAGE}")
message(STATUS "ADVANCED_DEBUGPROF: ${ADVANCED_DEBUGPROF}")
message(STATUS "ROCM_SYSTEM_INCLUDE: ${ROCM_SYSTEM_INCLUDE}")

if(WIN32)
    set(ROCM_SDK "$ENV(HIP_PATH)" CACHE PATH "HIP / ROCm SDK Dir")
    # https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/saxpy.html#setting-up-the-command-line
    find_program(ROC_DUMPBIN dumpbin) # TODO: Test on windows
    # TODO: dumpbin.exe /nologo /section:.hip_fat /rawdata:8 .\saxpy.exe 
else()
    # not all distors install rocm to /opt/rocm!
    find_program(HIP_CONFIG hipconfig HINTS "/opt/rocm" "/usr/bin")
    if(NOT HIP_CONFIG)
        message(FATAL_ERROR "hipconfig not found!")
    endif()

    execute_process(
        COMMAND ${HIP_CONFIG} --rocmpath OUTPUT_VARIABLE ROCM_SDK
    )

    find_program(ROC_OBJ roc-obj HINTS ${HIP_CONFIG})
    message(STATUS "roc-obj path: ${ROC_OBJ}")
endif()

message(STATUS "ROCm SDK: ${ROCM_SDK}")

# https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html
# https://gitlab.kitware.com/cmake/cmake/-/issues/24562
# https://discourse.cmake.org/t/cmake-hip-compiler-rocm-root-is-not-set-correctly/13102
# https://github.com/pytorch/pytorch/issues/128313#issuecomment-2167404020
# https://github.com/ROCm/HIP/issues/3406
# https://lists.llvm.org/pipermail/cfe-commits/Week-of-Mon-20250106/660706.html
# https://cgmb-rocm-docs.readthedocs.io/en/latest/Installation_Guide/Using-CMake-with-AMD-ROCm.html
# https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html
# https://github.com/pelagos-consulting/HIP_Course/blob/main/course_material/L4_Debugging/CMakeLists.txt

set(CMAKE_HIP_COMPILER_ROCM_ROOT ${ROCM_SDK}) # in case HIP clang is in non-standard location
set(CMAKE_${GPU_LANGUAGE}_STANDARD 17)
set(CMAKE_${GPU_LANGUAGE}_STANDARD_REQUIRED ON)
set(CMAKE_${GPU_LANGUAGE}_EXTENSIONS OFF)

# project needs to be defined before language is enabled below
project(
    fabsu
    VERSION 1.0
    DESCRIPTION "Fabsu"
    LANGUAGES ${PROJECT_LANGUAGE}) # when using ${GPU_LANGUAGE} or HIP here, VS Code syntax highlighting does not work

list(APPEND CMAKE_PREFIX_PATH "${ROCM_SDK}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

enable_language(${GPU_LANGUAGE})

# until CMake 4.1.0 we need to override the HIP debug flags (see https://gitlab.kitware.com/cmake/cmake/-/issues/26823)
set(CMAKE_HIP_FLAGS_DEBUG "-O0 -ggdb3") 

if(ADVANCED_DEBUGPROF)
    # dump assembly and intermediate files
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -save-temps")
    # print register/resource allocation per function
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage")
endif()

if(ROCM_SYSTEM_INCLUDE)
    # https://gitlab.kitware.com/cmake/cmake/-/issues/25346 help VSCode IntelilSense find <hip/hip_runtime.h>
    # This seems to work only in the Dev Container / Ubuntu case
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem ${ROCM_SDK}/include")

    # If you're on Fedora, like me, adding /usr/include to system include paths might lead to the following error:
    # [build] /usr/lib/gcc/x86_64-redhat-linux/14/../../../../include/c++/14/cmath:47:15: fatal error: 'math.h' file not found
    # [build]    47 | #include_next <math.h>
endif(ROCM_SYSTEM_INCLUDE)

message(STATUS "CMAKE_HIP_FLAGS_DEBUG: ${CMAKE_HIP_FLAGS_DEBUG}")
message(STATUS "CMAKE_HIP_FLAGS: ${CMAKE_HIP_FLAGS}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CMAKE_HIP_ARCHITECTURES:" ${CMAKE_HIP_ARCHITECTURES})

set(fabsu_app_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fabsu.cpp
)

add_executable(fabsu_app ${fabsu_app_SRC})

if(PROJECT_LANGUAGE STREQUAL "CXX")
    find_package(hip REQUIRED) # needed when project lang is CXX
    target_link_libraries(fabsu_app PRIVATE hip::host)
endif()

set_target_properties(fabsu_app PROPERTIES LINKER_LANGUAGE ${GPU_LANGUAGE})
set_source_files_properties(${fabsu_app_SRC} PROPERTIES LANGUAGE ${GPU_LANGUAGE})

target_include_directories(fabsu_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# target bar is only build when `make bar` is issued
# add_custom_target(bar
#     # cmake -E support copy/env/echo and so on. use cmake -E to see
#     # COMMAND/COMMENT must be upper case
#     # COMMAND ${CMAKE_COMMAND} -E echo bar:hello
#     # COMMAND ${CMAKE_COMMAND} -E environment

#     COMMENT "testing add_custom_target 'bar'..."
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}

#     #DEPENDS zoo
# )

# Assemble the device assemblies to object files using the HIP compiler.
# The compiler needs -target amdgcn-amd-amdhsa -mcpu=gfx* in order to assemble the object file
# for the right GPU.
# foreach(HIP_ARCHITECTURE ${CMAKE_HIP_ARCHITECTURES})
#     add_custom_command(
#         TARGET bar
#         # OUTPUT main_${HIP_ARCHITECTURE}.${OBJ_TYPE}
#         COMMAND
#             ${CMAKE_HIP_COMPILER} -fPIC -target amdgcn-amd-amdhsa
#             -mcpu=${HIP_ARCHITECTURE}
#             ${CMAKE_CURRENT_SOURCE_DIR}/main_${HIP_ARCHITECTURE}.s -o
#             ${CMAKE_CURRENT_BINARY_DIR}/main_${HIP_ARCHITECTURE}.${OBJ_TYPE}
#         DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/main_${HIP_ARCHITECTURE}.s
#         VERBATIM
#         COMMENT "Compiling ASM main_${HIP_ARCHITECTURE}.${OBJ_TYPE}"
#     )
# endforeach()

# execute_process(
#     COMMAND ${ROC_OBJ} ${CMAKE_CURRENT_BINARY_DIR}/main_${HIP_ARCHITECTURE}.${OBJ_TYPE}
# )