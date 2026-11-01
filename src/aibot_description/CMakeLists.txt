cmake_minimum_required(VERSION 3.16)
project(aibot_description)

find_package(ament_cmake REQUIRED)

# The model files are fetched and generated, not committed by hand. Fail early with a clear hint
# instead of ament's "can't find .../config" when the model files are missing.
foreach(generated
    LICENSE
    urdf/robot_original.urdf
    urdf/aibot_body.xacro
    config/joint_limits.yaml
    meshes/visual/base_link.STL
    meshes/collision/base_link.STL)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${generated}")
    message(FATAL_ERROR
      "aibot_description: missing ${generated}.\n"
      "Restore them with git from the workspace root, then rebuild.")
  endif()
endforeach()

install(
  DIRECTORY config launch meshes rviz urdf
  DESTINATION share/${PROJECT_NAME}
)
install(FILES LICENSE DESTINATION share/${PROJECT_NAME})

ament_package()
