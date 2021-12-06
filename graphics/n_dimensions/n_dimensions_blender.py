import bpy
import bmesh
from mathutils import Vector
import numpy as np
import math
import itertools

# Blender import system clutter
import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "graphics" / "n_dimensions"
sys.path.append(str(UTILS_PATH))
sys.path.append(str(SRC_PATH))

import n_dimensions_utils
import importlib
importlib.reload(n_dimensions_utils)

from n_dimensions_utils import get_simplex, imaging, subdivide, stereographic_projection,rotate
from n_dimensions_utils import get_hypercube, get_hyperoctahedron, get_24cell

### parameters are given in the call from animation-nodes

if polytope == 'simplex':
    points, edges, faces = get_simplex(n_dimensions, dist)
elif polytope == 'hypercube':
    points, edges, faces = get_hypercube(n_dimensions, dist)
elif polytope == 'hyperoctahedron':
    points, edges, faces = get_hyperoctahedron(n_dimensions, dist)
elif polytope == '24cell':
    points, edges, faces = get_24cell(n_dimensions, dist)
else:
    raise Exception('no such polytope: ' + polytope)
    
points, edges, faces = subdivide(points, edges, faces)
#points, edges, faces = subdivide(points, edges, faces)

### rotate object
if angle != 0.0:
    # make sure axis exists given the n-dimensions
    rot_axis1 = min(n_dimensions-1, rot_axis1)
    rot_axis2 = min(n_dimensions-1, rot_axis2)
    #points[:, rot_axis2] = points[:, rot_axis2] + shift
    points = rotate(points, angle, rot_axis1, rot_axis2)
if angle2 != 0.0:
    rot2_axis1 = min(n_dimensions-1, rot2_axis1)
    rot2_axis2 = min(n_dimensions-1, rot2_axis2)
    points = rotate(points, angle2, rot2_axis1, rot2_axis2)

points_scale = np.ones(len(points))
for idx in range(n_dimensions-3):
    points, norm_last_coord = stereographic_projection(points, 1+proj_fact*idx)
    points_scale = points_scale * (norm_last_coord+0.5)

# append 0 to obtain always at least 3D points (to make it work in Blender also for 1D and 2D)
if points.shape[-1] < 3:
    points = np.append(points, np.zeros([points.shape[0], 3-points.shape[-1]]), axis=-1)