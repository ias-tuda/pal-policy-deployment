cmake_minimum_required(VERSION 3.8)
project(pal_policy_inferer)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_executable(${PROJECT_NAME}_node 
src/inferer.cpp
src/onnx.cpp
)

ament_auto_add_executable(pal_policy_inferer_kangaroo_mjlab_monolithic_node
src/kangaroo_mjlab_monolithic_node.cpp
)

ament_auto_package()
