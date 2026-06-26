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
option(SA3_CUDA   "Build the GGML CUDA backend"   OFF)
option(SA3_VULKAN "Build the GGML Vulkan backend" OFF)
option(SA3_METAL  "Build the GGML Metal backend"  OFF)

if(SA3_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(SA3_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(SA3_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
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
