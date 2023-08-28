cmake_minimum_required(VERSION 3.22)
project(RangANN)

include_directories(include)
include_directories(include/hnswlib)
include_directories(include/ivf)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()
#添加头文件搜索路径
include_directories(D:/software/boost_1_78_0/)
include_directories(D:/software/eigen-3.4.0)
#/添加库文件搜索路径
link_directories(D:/software/boost_1_78_0/libs)

add_definitions (-std=c++11 -Ofast -lboost -march=native -Wall -DINFO)

add_executable(main ./src/main.cpp)


