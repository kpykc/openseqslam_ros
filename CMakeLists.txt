cmake_minimum_required(VERSION 2.8) 

PROJECT (openseqslam_ros)


find_package(catkin REQUIRED COMPONENTS 
				geometry_msgs
                cv_bridge 
                image_geometry
				image_transport 
				roscpp 
				sensor_msgs 
				std_msgs
                tf)


find_package(OpenCV REQUIRED )
#find_package(OpenCV REQUIRED core highgui imgproc features2d )

catkin_package(
  CATKIN_DEPENDS cv_bridge image_geometry image_transport roscpp sensor_msgs geometry_msgs
  DEPENDS system_lib OpenCV
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

include_directories(SYSTEM ${catkin_INCLUDE_DIRS})
include_directories(include)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")


set( NAME_SRC
    src/main.cpp src/OpenSeqSLAM.cpp
)

#set( NAME_HEADERS
#    OpenSeqSLAM.h
#)

add_executable( ${PROJECT_NAME} ${NAME_SRC} )
target_link_libraries( ${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS} )

install(TARGETS ${PROJECT_NAME}
        DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)