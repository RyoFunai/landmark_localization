cmake_minimum_required(VERSION 3.8)
project(landmark_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(landmark_localization
  src/landmark_localization.cpp
  src/pose_fuser.cpp
  src/ransac.cpp
)

ament_export_include_directories(include)
ament_export_libraries(landmark_localization)

ament_auto_package(INSTALL_TO_SHARE)