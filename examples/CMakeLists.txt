cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER acpp)
endif()

project(MiniGCP_examples CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -O3)

function(add_example name)
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE ../include)
endfunction()

add_example(NQueens)
add_example(TSP)