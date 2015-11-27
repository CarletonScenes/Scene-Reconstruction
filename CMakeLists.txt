cmake_minimum_required(VERSION 2.8)

find_package( OpenCV 3 REQUIRED )
# add_library( SC ${LIB_SOURCES} ${LIB_HEADERS} )

project(Tutorial)

# Ensure C++ 11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#add all subdirectories in this project
add_subdirectory(lib)
add_subdirectory(src)



# add_executable( GenCloud GenCloud.cpp )
# target_link_libraries( GenCloud SC )
# target_link_libraries( GenCloud ${OpenCV_LIBS} )



# cmake_minimum_required(VERSION 2.8)
# project( SceneComps )
# find_package( OpenCV 3 REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )
# file( GLOB LIB_SOURCES *.cpp )
# file( GLOB LIB_HEADERS *.h )
# add_library( SC ${LIB_SOURCES} ${LIB_HEADERS} )

# add_executable( FMTest FMTest.cpp)
# target_link_libraries( FMTest SC )
# target_link_libraries( FMTest ${OpenCV_LIBS} )

# add_executable( FDTest FDTest.cpp)
# target_link_libraries( FDTest SC )
# target_link_libraries( FDTest ${OpenCV_LIBS} )



# include_directories( easyexif )
# add_library(exif easyexif/exif.h easyexif/exif.cpp )
# add_executable( GenCloud GenCloud.cpp )
# target_link_libraries( GenCloud SC )
# target_link_libraries( GenCloud ${OpenCV_LIBS} )
