cmake_minimum_required(VERSION 3.20)
project(sa3_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Put our binaries next to the ggml DLLs so tools run without PATH juggling.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Backend toggles (forwarded to the vanilla ggml submodule).
# CUDA is our first-class target; others are end-stage goals.
option(SA3_CUDA   "Build the GGML CUDA backend"        OFF)
option(SA3_VULKAN "Build the GGML Vulkan backend"      OFF)
option(SA3_METAL  "Build the GGML Metal backend"       OFF)
option(SA3_HIP    "Build the GGML HIP/ROCm backend"    OFF)

if(SA3_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    # CUDA graphs need Ampere+ (CC >= 8.0). On older cards a stale GGML_CUDA_GRAPHS=ON
    # compiles graph code that prints "disabling CUDA graphs" at runtime. Force it OFF only
    # when we can PROVE the build targets exclusively pre-Ampere GPUs: every entry of
    # CMAKE_CUDA_ARCHITECTURES is a plain number < 80. Any non-numeric token (native, all,
    # all-major) means the arch is unknown at configure time, and any entry >= 80 means an
    # Ampere+ target is present, so in both cases we leave the ggml default untouched (this
    # is why -DCMAKE_CUDA_ARCHITECTURES=native must NOT disable graphs on capable hardware).
    if(CMAKE_CUDA_ARCHITECTURES)
        set(_sa3_all_pre_ampere ON)
        set(_sa3_any_numeric OFF)
        foreach(_arch ${CMAKE_CUDA_ARCHITECTURES})
            string(REGEX REPLACE "-(real|virtual)$" "" _arch_num "${_arch}")
            if(_arch_num MATCHES "^[0-9]+$")
                set(_sa3_any_numeric ON)
                if(_arch_num GREATER_EQUAL 80)
                    set(_sa3_all_pre_ampere OFF)
                    break()
                endif()
            else()
                set(_sa3_all_pre_ampere OFF)   # native/all/unknown -> can't prove pre-Ampere
                break()
            endif()
        endforeach()
        if(_sa3_any_numeric AND _sa3_all_pre_ampere)
            set(GGML_CUDA_GRAPHS OFF CACHE BOOL "" FORCE)
            message(STATUS "SA3_CUDA: CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} is all pre-Ampere; forcing GGML_CUDA_GRAPHS=OFF")
        endif()
    endif()
endif()
if(SA3_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(SA3_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()
if(SA3_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
endif()

# Vanilla GGML (no custom ops needed — the whole SA3 stack is attention/linear/norm/conv1d).
add_subdirectory(ggml)

# src/ is header-driven (acestep style); add library/tool targets here as phases land.
add_library(sa3 INTERFACE)
target_include_directories(sa3 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(sa3 INTERFACE ggml)

# --- tools -----------------------------------------------------------------
add_executable(sa3-smoke tools/sa3-smoke.cpp)
target_link_libraries(sa3-smoke PRIVATE sa3)

add_executable(sa3-codec tools/sa3-codec.cpp)
target_link_libraries(sa3-codec PRIVATE sa3)

add_executable(sa3-textenc tools/sa3-textenc.cpp)
target_link_libraries(sa3-textenc PRIVATE sa3)

add_executable(sa3-dit tools/sa3-dit.cpp)
target_link_libraries(sa3-dit PRIVATE sa3)

add_executable(sa3-textmusic tools/sa3-textmusic.cpp)
target_link_libraries(sa3-textmusic PRIVATE sa3)

add_executable(sa3-tokenize tools/sa3-tokenize.cpp)
target_link_libraries(sa3-tokenize PRIVATE sa3)

add_executable(sa3-generate tools/sa3-generate.cpp)
target_link_libraries(sa3-generate PRIVATE sa3)

# sa3-server: HTTP over the pipeline. Vendored cpp-httplib (split impl) + yyjson (C).
find_package(Threads REQUIRED)
add_executable(sa3-server tools/sa3-server.cpp
    vendor/cpp-httplib/httplib.cpp
    vendor/yyjson/yyjson.c)
target_link_libraries(sa3-server PRIVATE sa3 Threads::Threads)
target_include_directories(sa3-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/cpp-httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yyjson)
if(MSVC)
    set_source_files_properties(vendor/yyjson/yyjson.c vendor/cpp-httplib/httplib.cpp PROPERTIES COMPILE_FLAGS /w)
else()
    set_source_files_properties(vendor/yyjson/yyjson.c vendor/cpp-httplib/httplib.cpp PROPERTIES COMPILE_FLAGS -w)
endif()

# sa3-lora-convert: native safetensors->gguf LoRA conversion (no Python; src/lora_convert.h). Uses yyjson.
add_executable(sa3-lora-convert tools/sa3-lora-convert.cpp vendor/yyjson/yyjson.c)
target_link_libraries(sa3-lora-convert PRIVATE sa3)
target_include_directories(sa3-lora-convert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yyjson)

# --- libsa3: C ABI shared library (embed sa3 in a host app; see src/libsa3.h) ---------------
# Windows: sa3.dll + sa3.lib (import lib).  Unix/macOS: libsa3.so / libsa3.dylib.
add_library(sa3_shared SHARED src/libsa3.cpp vendor/yyjson/yyjson.c)
set_target_properties(sa3_shared PROPERTIES OUTPUT_NAME sa3)
target_link_libraries(sa3_shared PRIVATE sa3)
target_include_directories(sa3_shared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yyjson)

# Pure-C smoke test / usage example; links the import lib (proves the C ABI + boundary).
add_executable(sa3-libtest tools/sa3-libtest.c)
target_include_directories(sa3-libtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(sa3-libtest PRIVATE sa3_shared)

# C ABI cooperative-cancel smoke test for embedded hosts.
add_executable(sa3-libcancel tools/sa3-libcancel.c)
target_include_directories(sa3-libcancel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(sa3-libcancel PRIVATE sa3_shared)
