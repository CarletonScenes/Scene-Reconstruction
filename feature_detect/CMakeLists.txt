
include_directories($(CMAKE_CURRENT_SOURCE_DIR)/hdr)
cmake_minimum_required(VERSION 2.8)
project( SceneComps )
find_package( OpenCV 3 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
file( GLOB LIB_SOURCES *.cpp )
file( GLOB LIB_HEADERS *.h )
add_library( SC ${LIB_SOURCES} ${LIB_HEADERS} )

add_executable( FMTest FMTest.cpp)
target_link_libraries( FMTest SC )
target_link_libraries( FMTest ${OpenCV_LIBS} )

add_executable( FDTest FDTest.cpp)
target_link_libraries( FDTest SC )
target_link_libraries( FDTest ${OpenCV_LIBS} )



# include_directories( easyexif )
# add_library(exif easyexif/exif.h easyexif/exif.cpp )
# add_executable( GenCloud GenCloud.cpp )
# target_link_libraries( GenCloud SC )
# target_link_libraries( GenCloud ${OpenCV_LIBS} )
