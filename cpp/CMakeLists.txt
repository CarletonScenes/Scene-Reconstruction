cmake_minimum_required(VERSION 2.8)
project( SceneComps CXX )
find_package( OpenCV 3 REQUIRED )

# Ensure C++11 flag set in compiler.
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
# if(COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# elseif(COMPILER_SUPPORTS_CXX0X)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
# else()
#     message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
# endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#add all subdirectories in this project
include_directories( ${PROJECT_SOURCE_DIR}/cpp/lib )
add_subdirectory(cpp/src)

