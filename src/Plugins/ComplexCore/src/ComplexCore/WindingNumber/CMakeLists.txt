cmake_minimum_required(VERSION 3.26)
project(WindingNumber
  VERSION 0.1.0
  DESCRIPTION "Winding Number"
  HOMEPAGE_URL "ssh://git@github.com/alecjacobson/WindingNumber"
  LANGUAGES CXX
)


set(COMPLEX_BIN_DIR ${PROJECT_BINARY_DIR}/bin)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${COMPLEX_BIN_DIR} CACHE PATH "Single Directory for all Libraries")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${COMPLEX_BIN_DIR} CACHE PATH "Single Directory for all Executables.")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${COMPLEX_BIN_DIR} CACHE PATH "Single Directory for all static libraries.")

list(APPEND CMAKE_MODULE_PATH ${complex_SOURCE_DIR}/cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(TBB CONFIG REQUIRED)


add_executable(fastwinding main.cpp UT_SolidAngle.cpp UT_Array.cpp)
target_link_libraries(fastwinding
Eigen3::Eigen
TBB::tbb
)

