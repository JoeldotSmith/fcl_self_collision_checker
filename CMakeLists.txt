cmake_minimum_required(VERSION 3.5)
project(fcl_self_collision_checker)

set(CMAKE_PREFIX_PATH "/usr/local")  # Ensure CMake looks here first
set(FCL_DIR "/usr/local/lib/cmake/fcl")

find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(urdfdom REQUIRED)
find_package(urdf REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(fcl REQUIRED)
find_package(ament_index_cpp REQUIRED)

add_executable(collision_checker src/main.cpp)

target_include_directories(collision_checker PUBLIC
  /usr/local/include  # FCL headers
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(collision_checker
  fcl
)


ament_target_dependencies(collision_checker fcl rclcpp urdfdom sensor_msgs urdf ament_index_cpp kdl_parser)

install(
  DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS collision_checker
  DESTINATION lib/${PROJECT_NAME})

ament_package()