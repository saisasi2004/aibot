# MediaPipe Hand landmarks (21 points) of ONE operator hand for one camera frame.
#
# 0 wrist | thumb 1 CMC, 2 MCP, 3 IP, 4 TIP | index 5 MCP, 6 PIP, 7 DIP, 8 TIP
# middle 9-12 | ring 13-16 | pinky 17-20   (same MCP, PIP, DIP, TIP order)

std_msgs/Header header        # stamp = capture time of the camera frame (same as BodyLandmarks)
string side                   # "left" or "right": the operator's anatomical hand
bool detected                 # false: this hand is not tracked in this frame and arrays are empty
float32 score                 # 0..1 hand detection / handedness confidence

geometry_msgs/Point[] image_points   # normalised raw-image coordinates, as in BodyLandmarks
geometry_msgs/Point[] world_points   # metres, origin at the hand's centre, camera-aligned axes
