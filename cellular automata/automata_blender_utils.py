import bpy
from mathutils import Vector
import math
import numpy as np

#######################################
#           GENERATORS                #
#######################################
# generate and add an object with given properties to the scene


def suzanne_generator(size, x, y, z):
    bpy.ops.mesh.primitive_monkey_add(
                        radius=size,
                        location=(x*2, y*2, z*2))


def cube_generator(cube_side, x, y, z):
    bpy.ops.mesh.primitive_cube_add(
                        size=cube_side,
                        location=(x*2, y*2, z*2))


def icosphere_generator(size, subdivisions, x, y, z):
    bpy.ops.mesh.primitive_ico_sphere_add(
                        subdivisions=subdivisions,
                        radius=size,
                        location=(x*2, y*2, z*2))


#######################################
#             UPDATERS                #
#######################################
# define behavior for a Blender object based on gol grid value

# Hides object (both view and render)
def object_updater_hide(obj, grid_val, keyframe=True):
    obj.hide_viewport = not grid_val
    obj.hide_render = obj.hide_viewport
    if keyframe:
        obj.keyframe_insert("hide_viewport")
        obj.keyframe_insert("hide_render")


# shrink object when grid values is zero
def object_updater_scale(obj, grid_val, scale_factor=0.8, keyframe=True):
    origin_scale = Vector((1.0, 1.0, 1.0))
    # grid value 1, object should end up with original size
    if grid_val:
        # skip all (keyframing too) if already ok, otherwise set original size
        if obj.scale == origin_scale:
            return
        else:
            obj.scale = origin_scale
    # grid value 0, object should end up scaled
    else:
        # skip all (keyframing too) if already ok, otherwise set scaled size
        if obj.scale == origin_scale*scale_factor:
            return
        else:
            obj.scale = origin_scale*scale_factor
    if keyframe:
        obj.keyframe_insert("scale")


# change color of obj based on given values
def object_updater_color_vector(obj, grid_vals, keyframe=True):
    mat = obj.active_material
    mat.diffuse_color = bin_to_color(grid_vals)
    if keyframe:
        mat.keyframe_insert("diffuse_color")

#######################################
#           GENERAL METHODS           #
#######################################


# create grid of objects on current scene
# The object generator is responsible for the creation of a single object instance
def create_grid(automaton, obj_generator):
    obj_grid = np.empty(automaton.grid.shape, dtype=object)
    for idx, val in np.ndenumerate(automaton.grid):
        # does not return object, so need to select active one
        obj_generator(idx)
        obj_grid[idx] = bpy.context.view_layer.objects.active
    return obj_grid


# update grid of Blender objects to reflect automaton status
def update_grid(obj_grid, automaton, obj_updater):
    for idx, val in np.ndenumerate(automaton.grid):
        obj_updater(obj_grid[idx], automaton.grid, idx)

#######################################
#            UTIL METHODS             #
#######################################

# convert a binary array into % RBG values
# the idea is to adapt to any array size and allowing len(array)/3
# bits per color (BPC). [R*BPC G*BPC B*BPC]
# Color value is then computed as
# binary code number divided by the max representable code number
def bin_to_color(v):
    if len(v)<3:
        return 0.5,0.5,0.5
    bpc = len(v)//3 #bit-per-color
    # max representable code number
    max_code = int("".join(['1']*bpc), 2)
    r = int("".join(map(str, v[0:bpc])), 2)/max_code
    g = int("".join(map(str, v[bpc:bpc*2])), 2)/max_code
    b = int("".join(map(str, v[bpc*2:bpc*3])), 2)/max_code
    return r,g,b


# for all elements in the grid, create and set a new material
def init_materials(obj_grid, init_color):
    for index, obj in np.ndenumerate(obj_grid):
        mat = bpy.data.materials.new("mat_{}".format(index))
        mat.diffuse_color = init_color
        obj.active_material = mat


def calculate_hexagonal_cell_position(row, col, nb_rows, nb_cols, size):
    # Hexagonal shape size for grid adjustment
    hex_size = size * math.cos(math.pi / 6)
    short_size = size / 2

    # Calculate row and col position for the current cell
    # taking into account hexagonal shape and shifting by growth
    y = (row - nb_rows // 2) * (2 * size - short_size)
    x = (col - nb_cols // 2) * (2 * hex_size) - hex_size
    # shift even rows
    if row % 2 == 0:
        x += hex_size

    return y, x


# Dummy method to load some good configs from the exploratory generations
def load_good_configs(dir):
    imgs = dir.glob('*.png')
    good_runs = [int(img.stem.split('_')[1]) for img in imgs]
    confs = []
    with open(dir / 'logs.txt') as f:
        for i, line in enumerate(f):
            if i in good_runs:
                print(i)
                p_freeze, p_melt = line.split('-')
                p_freeze = list(map(float, p_freeze.split(':')[1][1:-2].split(' ')))
                p_melt = list(map(float, p_melt.split(':')[1][1:-2].split(' ')))
                confs.append((p_freeze, p_melt))
    print(len(confs))
    print(confs[0])
    return confs
