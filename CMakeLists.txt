cmake_minimum_required(VERSION 3.22)
project(RangANN)

include_directories(include)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

add_definitions(-std=c++17 -Ofast -march=core-avx2 -mavx512f -fpic -fopenmp -ftree-vectorize -fexceptions)

add_subdirectory(src)

