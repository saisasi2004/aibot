# One object the robot perceives with its head camera (perception_node). World frame.
string id                       # stable across frames, e.g. "red_cube_1"
string label                    # cube | box | block | can | mug | ball | bin | tray | (detector noun)
string color
geometry_msgs/Pose pose         # centroid; orientation = yaw about z of the major footprint axis
geometry_msgs/Vector3 size      # x: major footprint extent, y: minor extent, z: height (m)
geometry_msgs/Pose grasp_pose   # top-face centre; orientation = yaw of the closing direction (minor axis)
float32 confidence
builtin_interfaces/Time last_seen
string held_by                  # "" | "left" | "right"
string inside                   # container id this object was put into ("" if none)
int32[4] box                    # image box (u0 v0 u1 v1, head camera pixels) of the last observation; zeros if none
string semantic_name            # what the VLM says it is ("coffee mug"); geometry above stays the detector's
string description              # short VLM description; both empty until labelled
