cmake_minimum_required(VERSION 2.8)

project( Depth )

find_package( OpenCV REQUIRED )
find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable( DepthMap depth.cpp )
target_link_libraries( DepthMap ${OpenCV_LIBS} )
target_link_libraries(DepthMap ${PCL_LIBRARIES})

add_executable( SGBMathcing sgbm_matching.cpp )
target_link_libraries( SGBMathcing ${OpenCV_LIBS} )


