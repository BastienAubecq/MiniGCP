cmake_minimum_required(VERSION 3.20)
project(MiniGCP_tests CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER acpp)
add_compile_options(-Wall)

find_package(Catch2 3 REQUIRED)

function(add_test_file target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
    target_link_libraries(${target} PRIVATE Catch2::Catch2WithMain)
    catch_discover_tests(${target})
endfunction()

include(CTest)
include(Catch)

add_test_file(test_AllDifferentFWC constraints/test_AllDifferentFWC.cpp)
add_test_file(test_NotEqual constraints/test_NotEqual.cpp)
add_test_file(test_Sum constraints/test_Sum.cpp)
add_test_file(test_FirstFail search/test_FirstFail.cpp)
add_test_file(test_FixRemove search/test_FixRemove.cpp)
add_test_file(test_AtomicIntVarRef state/test_AtomicIntVarRef.cpp)
add_test_file(test_IntVarRef state/test_IntVarRef.cpp)