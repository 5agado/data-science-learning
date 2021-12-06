from math import cos, sin, pi, exp, sqrt
import numpy as np

positions = []
rotations = []
scales = []
shift_factor = np.array(shift_factor)

# TODO spiral types are applied only to shift factor while positions are purely based on circular angle.
# So the radius-shift doesn't appropriately results in the actual spiral type
spiral_types = ['archimedean', 'hyperbolic', 'fermat', 'lituus', 'log']

if spiral_type not in spiral_types:
    spiral_type = 'archimedean'

# iterate through each element
for i in range(nb):
    # angle increases constantly for each new element
    angle = angle_shift * i  # angle is in radians
    
    # transformation shift
    if spiral_type == 'archimedean':
        trans_shift = shift_factor * (angle/(2*pi))
    elif spiral_type == 'log':
        trans_shift = shift_factor * exp(angle/(2*pi))
    elif spiral_type == 'hyperbolic':
        trans_shift = shift_factor / (angle/(2*pi))
    elif spiral_type == 'fermat':
        trans_shift = shift_factor * sqrt(angle/(2*pi))
    elif spiral_type == 'lituus':
        trans_shift = shift_factor / sqrt(angle/(2*pi))
    
    # radius, scale and rotation 
    # add to starting value the given shift proportional to the angle
    radius = np.array(radius_base) + (radius_shift * trans_shift[0])
    scale = np.array(scale_base) + (scale_shift * trans_shift[1])
    rotation = np.array(rotation_base) + (rotation_shift * trans_shift[2])
    
    # find position based on radius and angle
    x = center[0] + radius[0] * cos(angle)
    y = center[1] + radius[1] * sin(angle)
    z = center[2] - radius[2]    

    positions.append((x, y, z))
    # add rotation on z such that element always points outward
    rotations.append((rotation[0], rotation[1], rotation[2]+ angle%(2*pi)))
    scales.append(list(scale))