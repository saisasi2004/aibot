# MediaPipe Pose (BlazePose, 33 points) of the operator for one camera frame.
#
# Index order is MediaPipe's: 0 nose, 11/12 shoulders, 13/14 elbows, 15/16 wrists,
# 17-22 pinky/index/thumb knuckles, 23/24 hips, 25-32 legs. Odd = operator's LEFT side,
# even = operator's RIGHT side (anatomical, as BlazePose reports them on the raw image).

std_msgs/Header header        # stamp = capture time of the camera frame
bool detected                 # false: nobody tracked in this frame and all arrays are empty

# Normalised coordinates in the RAW (un-mirrored) camera image: x in [0,1] to the right,
# y in [0,1] downwards, z = depth relative to the hip midpoint (MediaPipe's image-width scale).
geometry_msgs/Point[] image_points

# Metric 3D estimate in metres, origin at the hip midpoint, camera-aligned axes:
# x = image right, y = image down, z = away from the camera.
geometry_msgs/Point[] world_points

float32[] visibility          # 0..1 likelihood the landmark is visible (not occluded)
float32[] presence            # 0..1 likelihood the landmark is inside the frame

uint32 image_width
uint32 image_height
