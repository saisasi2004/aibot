std_msgs/Header header
aibot_msgs/DetectedObject[] objects
string detector                 # tabletop | yolo_world | color_blob
float32 detect_ms               # detector (2D or segmentation) time for the last frame
float32 fusion_ms               # depth fusion / 3D estimation time for the last frame
