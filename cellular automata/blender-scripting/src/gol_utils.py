import bpy
from mathutils import Vector
import numpy as np
import configparser

#######################################
#           GENERATORS                #
#######################################
# generate and add an object with given properties to the scene

def suzanne_generator(size, x, y, z):
    bpy.ops.mesh.primitive_monkey_add(
                        radius = size,
                        location = (x*2, y*2, z*2))

def cube_generator(cube_side, x, y, z):
    bpy.ops.mesh.primitive_cube_add(
                        radius = cube_side,
                        location = (x*2, y*2, z*2))

def icosphere_generator(size, subdivisions, x, y, z):
    bpy.ops.mesh.primitive_ico_sphere_add(
                        subdivisions = subdivisions,
                        radius = size,
                        location = (x*2, y*2, z*2))


#######################################
#             UPDATERS                #
#######################################
# define behavior for a Blender object based on gol grid value

# Hides object (both view and render)
def object_updater_hide(obj, grid_val, keyframe=True):
    obj.hide = not grid_val
    obj.hide_render = obj.hide
    if keyframe:
        obj.keyframe_insert("hide")
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

# delete all objects of current scene (un-hide all hidden ones first)
def delete_all():
    for obj in bpy.data.objects:
        obj.hide = False
        # select/delete only meshes
        obj.select = obj.type == 'MESH'
    bpy.ops.object.delete(use_global=True)


def load_GOL_config(config_path, section):
    config = configparser.ConfigParser()
    config.read(config_path)
    GOL_config = {}
    entries = ['neighbours_count_born', 'neighbours_maxcount_survive', 'neighbours_mincount_survive']
    for e in entries:
        GOL_config[e] = config.getint(section, e)
    return GOL_config