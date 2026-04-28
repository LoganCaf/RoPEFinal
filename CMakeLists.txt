cmake_minimum_required(VERSION 3.16)

project(RoPESerialBaseline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(rope_serial
  src/attention.cpp
  src/matrix.cpp
  src/metrics.cpp
  src/rotary_embedding.cpp
)

target_include_directories(rope_serial
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(rope_serial PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

add_executable(rope_benchmark benchmarks/rope_benchmark.cpp)
target_link_libraries(rope_benchmark PRIVATE rope_serial)

enable_testing()

add_executable(rope_tests tests/test_serial.cpp)
target_link_libraries(rope_tests PRIVATE rope_serial)
add_test(NAME rope_tests COMMAND rope_tests)

