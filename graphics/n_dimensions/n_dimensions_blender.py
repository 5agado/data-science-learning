import bpy
import numpy as np
from enum import Enum

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

from n_dimensions_utils import get_simplex, get_hypercube, get_hyperoctahedron, get_24cell, get_600cell

class Polytope(Enum):
    simplex = 1
    hypercube = 2
    hyperoctahedron = 3
    cell24 = 4
    cell600 = 5

def create_mesh(name, col_name, n):
    verts, edges, faces = get_simplex(n)

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections[col_name]
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(verts[:, :3], edges, faces)

    if n > 3:
        for dim in range(3, n):
            attr_name = f'{dim + 1}_dim'
            obj.data.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
            obj.data.attributes[attr_name].data.foreach_set('value', verts[:, dim])

def main(polytope: Polytope, n_dimensions: int, obj_name: str, collection_name: str, dist=1.):
    if polytope == Polytope.simplex:
        points, edges, faces = get_simplex(n_dimensions, dist)
    elif polytope == Polytope.hypercube:
        points, edges, faces = get_hypercube(n_dimensions, dist)
    elif polytope == polytope.hyperoctahedron:
        points, edges, faces = get_hyperoctahedron(n_dimensions, dist)
    elif polytope == polytope.cell24:
        points, edges, faces = get_24cell(n_dimensions, dist)
    elif polytope == polytope.cell600:
        points, edges, faces = get_600cell(n_dimensions, dist)
    else:
        raise Exception('no such polytope: ' + polytope.name)

    #points, edges, faces = subdivide(points, edges, faces)

    # append 0 to obtain always at least 3D points (to make it work in Blender also for 1D and 2D)
    if points.shape[-1] < 3:
        points = np.append(points, np.zeros([points.shape[0], 3-points.shape[-1]]), axis=-1)

    # create mesh
    mesh = bpy.data.meshes.new(obj_name)
    obj = bpy.data.objects.new(mesh.name, mesh)
    if collection_name not in bpy.data.collections:
        col = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(col)
    else:
        col = bpy.data.collections[collection_name]
    col.objects.link(obj)
    mesh.from_pydata(points[:, :3], edges, faces)

    # store each coordinate as GN attribute
    for dim in range(0, n_dimensions):
        attr_name = f'{dim + 1}_dim'
        obj.data.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
        obj.data.attributes[attr_name].data.foreach_set('value', points[:, dim])


for n_dim in range(2, 5):
    target_polytope = Polytope.hypercube
    main(target_polytope, n_dim, f'{target_polytope.name}_{n_dim}', target_polytope.name)