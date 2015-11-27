#name the executable and all resources it depends on directly
add_executable( GenCloud GenCloud.cpp )
#link to needed libraries
target_link_libraries(GenCloud exif)
#tell CMake where to look for the .h files
target_link_libraries( GenCloud SC )
target_link_libraries( GenCloud ${OpenCV_LIBS} )
target_include_directories( GenCloud PUBLIC ${CMAKE_SOURCE_DIR}/lib)
