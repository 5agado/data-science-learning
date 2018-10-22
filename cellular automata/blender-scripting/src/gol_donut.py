# Blender import system clutter
import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "cellular automata/blender-scripting"
sys.path.append(str(SRC_PATH))
sys.path.append(str(UTILS_PATH))

CONFIG_PATH = str(SRC_PATH / 'GOL_config.ini')

import utils.blender_utils
import importlib
importlib.reload(utils.blender_utils)

from utils.blender_utils import delete_all
from mathutils import Matrix, Vector
import numpy as np
import bpy

import gol_utils as utils
from conway_2D import ConwayGOL_2D


# update grid of Blender objects to reflect gol status, then update gol.
def update_grid(obj_grid, gol, obj_updater):
    for i in range(gol.N):
        for j in range(gol.N):
            obj_updater(obj_grid[i][j], gol.grid, (i, j))
    gol.update()


def object_updater_move(obj, grid_val):
    if grid_val:
        obj['obj'].location = obj['translated_loc']
    else:
        obj['obj'].location = obj['original_loc']
    obj['obj'].keyframe_insert("location")


def main():
    NUM_FRAMES_CHANGE = 5
    NUM_FRAMES = 50
    bpy.context.scene.frame_end = NUM_FRAMES

    torus_major_segments = 30
    torus_minor_segments = 15
    torus_major_radius = 1
    torus_minor_radius = 0.25
    obj_size = 0.05
    move_factor = 2

    grid_side = int(np.sqrt(torus_major_segments*torus_minor_segments))

    delete_all()

    utils.cube_generator(obj_size, 0, 0, 0)
    cube = bpy.context.scene.objects.active

    # create Donut
    bpy.ops.mesh.primitive_torus_add(location=(0, 0, 0), major_radius=torus_major_radius,
                                     minor_radius=torus_minor_radius, abso_major_rad=1., abso_minor_rad=1.,
                                     major_segments=torus_major_segments,
                                     minor_segments=torus_minor_segments)
    torus = bpy.context.scene.objects.active

    # duplicate object by donut faces
    cube.select = True
    torus.select = True
    bpy.ops.object.parent_set(type='VERTEX')
    bpy.context.object.dupli_type = 'FACES'
    bpy.ops.object.duplicates_make_real()

    obj_updater = lambda obj, grid, idx: object_updater_move(obj, grid[idx])

    # create GOL data
    gol = ConwayGOL_2D(grid_side,
                       utils.load_GOL_config(CONFIG_PATH, 'GOL_3D_standard'),
                       seed=42)
    # create GOL object grid by selected all previously created objects
    objs = [bpy.context.scene.objects["Cube.{:03d}".format(int(i))]
            for i in range(1, (grid_side*grid_side)+1)]

    # Add camera path and follow path action
    bpy.ops.curve.primitive_bezier_circle_add(radius=torus_major_radius, location=(0, 0, 0))
    cube.select = False
    torus.select = False
    bpy.data.objects['BezierCircle'].select = False
    camera = bpy.data.objects['Camera']
    camera.select = True
    camera.location = (0, 0, 0)
    camera.rotation_euler = (0, 0, 0)
    camera.constraints.new(type='FOLLOW_PATH')
    follow_path = camera.constraints['Follow Path']
    follow_path.target = bpy.data.objects["BezierCircle"]
    follow_path.forward_axis = 'TRACK_NEGATIVE_Z'
    follow_path.up_axis = 'UP_Y'
    follow_path.use_curve_follow = True
    #bpy.ops.constraint.followpath_path_animate(constraint="Follow Path", owner='OBJECT')
    bpy.data.objects['Camera'].rotation_euler = (0, -0.5, 0)

    # Set objects on and off location by translating across object axes
    obj_grid = []
    trans_local_adjust = Vector((0.0, 0.0, obj_size+obj_size/10))
    trans_local_move = Vector((0.0, 0.0, -obj_size*move_factor))
    for obj in objs:
        trans_world = obj.matrix_world.to_3x3() * trans_local_adjust
        obj.matrix_world.translation += trans_world
        original_loc = obj.location.copy()
        trans_world = obj.matrix_world.to_3x3() * trans_local_move
        obj.matrix_world.translation += trans_world
        translated_loc = obj.location.copy()
        obj_grid.append({'obj': obj, 'original_loc': original_loc,
                         'translated_loc': translated_loc})

    obj_grid = np.array(obj_grid).reshape((grid_side, grid_side))

    for frame in range(1, NUM_FRAMES+1):
        if frame % 10 == 0:
            print("Updating frame {}".format(frame))
        bpy.context.scene.frame_set(frame)
        # When reaching final frame, clear handlers
        if frame >= NUM_FRAMES:
            bpy.app.handlers.frame_change_pre.clear()
            bpy.context.scene.frame_set(1)
        elif (frame % NUM_FRAMES_CHANGE) == 0:
            update_grid(obj_grid, gol, obj_updater)


main()