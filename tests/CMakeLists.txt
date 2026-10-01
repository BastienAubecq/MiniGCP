cmake_minimum_required(VERSION 3.20)
project(MiniGCP_tests CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER acpp)
set(CMAKE_CXX_FLAGS -Wall)

find_package(Catch2 3 REQUIRED)

add_executable(tests
    state/test_IntVarRef.cpp
    state/test_AtomicIntVarRef.cpp
)

target_include_directories(tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(tests)