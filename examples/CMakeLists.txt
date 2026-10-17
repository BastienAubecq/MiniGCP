cmake_minimum_required(VERSION 3.20)
project(MiniGCP_examples CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER acpp)
add_compile_options(-Wall -O3)

function(add_example name)
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE ../include)
endfunction()

add_example(NQueens)
add_example(TSP)