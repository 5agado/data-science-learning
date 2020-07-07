from math import cos, sin, pi
import numpy as np

positions = []
rotations = []
scales = []

# iterate through each element
for i in range(nb):
    # angle increases constantly for each new element
    angle = angle_shift * i  # angle is in radians
    
    # radius, scale and rotation 
    # add to starting value the given shift multiplied by how many times we
    # completed a full round
    trans_shift = np.array(shift_factor) * (angle//(2*pi))
    radius = np.array(base_radius) + (radius_shift * trans_shift[0])
    scale = np.array(base_scale) + (scale_shift * trans_shift[1])
    rotation = np.array(base_rotation) + (rotation_shift * trans_shift[2])
    
    # find position based on radius and angle
    x = center[0] + radius[0] * cos(angle)
    y = center[1] + radius[1] * sin(angle)
    z = center[2] - radius[2]    

    positions.append((x, y, z))
    # add rotation on z such that element always points outward
    rotations.append((rotation[0], rotation[1], rotation[2]+ angle%(2*pi)))
    scales.append(list(scale))