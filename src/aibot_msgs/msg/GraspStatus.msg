# What each robot hand is holding (grasp_manager_node, /aibot/grasp/status).
# Reported from the gz DetachableJoint state, not from the command, so a skill can verify a grasp.
std_msgs/Header header
string left_object       # "" when empty
string right_object
bool ready               # false until every pair was detached after spawn
