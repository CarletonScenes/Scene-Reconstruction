include_directories( ${OpenCV_INCLUDE_DIRS} )

SET(LIB_SOURCES
    ../lib/exif.cpp
    ../lib/Image.cpp
)

SET(LIB_HEADERS
    ../lib/FeatureDetector.h
    ../lib/FeatureDetectorSIFT.h
    ../lib/Image.h
    ../lib/KeypointDescriptor.h
    ../lib/OpenCV.h
)

add_library( SC ${LIB_HEADERS} ${LIB_SOURCES} )
target_link_libraries( SC ${OpenCV_LIBS} )

#name the executable and all resources it depends on directly
add_executable( GenCloud GenCloud.cpp )
#link to needed libraries
target_link_libraries(GenCloud exif)

#tell CMake where to look for the .h files
MESSAGE(STATUS)

target_link_libraries( GenCloud ${OpenCV_LIBS} )
# target_include_directories( GenCloud PUBLIC ${CMAKE_SOURCE_DIR}/lib)
target_link_libraries( GenCloud SC )
