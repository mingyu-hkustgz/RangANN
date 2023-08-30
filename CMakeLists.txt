cmake_minimum_required(VERSION 3.22)
project(RangANN)

include_directories(include)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

add_definitions(-std=c++11 -O3 -lboost -Wall -DINFO)

add_executable(test_segment_hnsw src/test_segment_hnsw.cpp)

add_executable(test_segment_ivf src/test_segment_ivf.cpp)

add_executable(build_ivf_index src/build_ivf_index.cpp)



