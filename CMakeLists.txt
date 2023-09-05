cmake_minimum_required(VERSION 3.12)
project("sam.cpp")

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(SAM_STANDALONE ON)
else()
    set(SAM_STANDALONE OFF)
endif()

option(SAM_BUILD_EXAMPLES "sam: build examples" ${SAM_STANDALONE})

add_subdirectory(ggml)

set(SAM_LIB sam.cpp)

add_library(${SAM_LIB} sam.cpp sam.h)
target_link_libraries(${SAM_LIB} PUBLIC ggml)
target_include_directories(${SAM_LIB} PUBLIC .)
target_compile_features(${SAM_LIB} PUBLIC cxx_std_14)

if (SAM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
