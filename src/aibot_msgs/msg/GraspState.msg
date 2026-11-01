# Grasp intent of one robot hand, published by retarget_node and consumed by grasp_manager_node.

std_msgs/Header header
string side                   # "left" or "right": the ROBOT hand this state drives
float32 closure               # 0 (open) .. 1 (fist): mean finger curl after filtering
bool closed                   # closure thresholded with hysteresis: the grasp command
bool tracking                 # false: the operator's hand is not tracked, values are held
