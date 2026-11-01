cmake_minimum_required(VERSION 3.16)
project(aibot_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BodyLandmarks.msg"
  "msg/HandLandmarks.msg"
  "msg/GraspState.msg"
  "msg/GraspStatus.msg"
  "msg/PlaceTarget.msg"
  "msg/DetectedObject.msg"
  "msg/DetectedObjectArray.msg"
  "srv/SetMode.srv"
  "srv/ResetScene.srv"
  "srv/SetPrompts.srv"
  "srv/SetHeld.srv"
  "srv/VlmQuery.srv"
  "action/ExecuteSkill.action"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
