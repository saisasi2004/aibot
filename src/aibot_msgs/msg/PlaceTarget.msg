# Where to put an object, as the planner states it. Resolved to a pose by deterministic code
# (aibot_ai/geometry/place.py); the LLM never produces coordinates except an explicit at_xy.
string relation          # on | in | left_of | right_of | in_front_of | behind | at_region | at_xy
string reference_id      # object/container id for on, in, left_of, ...
string region            # at_region: left | right | front | back | center | front_left | ...
float64 x                # at_xy only, world frame (m)
float64 y
