cmake_minimum_required(VERSION 3.15)

project(ccro_nubs_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

set(NUBS_INCLUDE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../NUBSTrajectory-main/include"
    CACHE PATH "Directory containing NUBSTrajectory.hpp")

if(NOT EXISTS "${NUBS_INCLUDE_DIR}/NUBSTrajectory.hpp")
    message(FATAL_ERROR "NUBSTrajectory.hpp not found under ${NUBS_INCLUDE_DIR}")
endif()

pybind11_add_module(_nubs_cpp nubs_pybind.cpp)
target_include_directories(_nubs_cpp PRIVATE "${NUBS_INCLUDE_DIR}")
target_link_libraries(_nubs_cpp PRIVATE Eigen3::Eigen)
target_compile_options(_nubs_cpp PRIVATE -Wall -Wextra -Wpedantic)

set_target_properties(_nubs_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
)
