# Blender import system clutter
import sys
from pathlib import Path
from mathutils import Matrix, Vector
import bpy

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "cellular automata"
sys.path.append(str(SRC_PATH))
sys.path.append(str(UTILS_PATH))

import Automaton
import automata_blender_utils
import importlib
importlib.reload(Automaton)
importlib.reload(automata_blender_utils)
from Automaton import *

from ds_utils.blender_utils import delete_all


def object_updater_move(obj, grid_val):
    if grid_val:
        obj['obj'].location = obj['translated_loc']
    else:
        obj['obj'].location = obj['original_loc']
    obj['obj'].keyframe_insert("location")


def set_gol_to_glider(gol, row_shift=0, col_shift=0):
    gol.grid.fill(0)
    gol.grid[row_shift, col_shift + 1] = 1
    gol.grid[row_shift+1, col_shift + 2] = 1
    gol.grid[row_shift+2, col_shift:col_shift+3] = 1


def main():
    NUM_FRAMES_CHANGE = 5
    NUM_FRAMES = 160
    PATH_DURATION = 100
    MIRROR_FRAME = 80
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = NUM_FRAMES

    assert (NUM_FRAMES % NUM_FRAMES_CHANGE == 0)

    torus_major_segments = 70
    torus_minor_segments = 25
    torus_major_radius = 1
    torus_minor_radius = 0.25
    seed = 10
    obj_size = 0.015
    move_factor = 2

    delete_all()

    automata_blender_utils.cube_generator(obj_size, 0, 0, 0)
    cube = bpy.context.view_layer.objects.active

    print("-----------------------")
    print("Start creation process")

    # create Donut
    bpy.ops.mesh.primitive_torus_add(location=(0, 0, 0), major_radius=torus_major_radius,
                                     minor_radius=torus_minor_radius, abso_major_rad=1., abso_minor_rad=1.,
                                     major_segments=torus_major_segments,
                                     minor_segments=torus_minor_segments)
    torus = bpy.context.view_layer.objects.active

    # duplicate object by donut faces
    cube.select = True
    torus.select = True
    bpy.ops.object.parent_set(type='VERTEX')
    bpy.context.object.dupli_type = 'FACES'
    bpy.ops.object.duplicates_make_real()

    obj_updater = lambda obj, grid, idx: object_updater_move(obj, grid[idx])

    # create GOL data
    rule_gol = {'neighbours_count_born': 3,
                'neighbours_maxcount_survive': 3,
                'neighbours_mincount_survive': 2,
                }
    gol = Automaton2D(nb_rows=torus_major_segments, nb_cols=torus_minor_segments,
                      config=rule_gol,
                      seed=seed)
    gol.update()  # update gol to start with a cleaner grid
    gol_grid_cache = []

    # create GOL object grid by selected all previously created objects
    objs = [obj for name, obj in bpy.context.scene.objects.items() if name.startswith('Cube.')]
    assert len(objs) == gol.grid.shape[0] * gol.grid.shape[1]

    #set_gol_to_glider(gol)

    # Add camera path and follow path action
    bpy.ops.curve.primitive_bezier_circle_add(radius=torus_major_radius, location=(0, 0, 0))
    cube.select = False
    torus.select = False
    bpy.data.objects['BezierCircle'].select = False
    #bpy.data.objects['BezierCircle'].data.path_duration = PATH_DURATION
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

    # Animate path. Doesn't seem to work because of wrong context
    #bpy.ops.constraint.followpath_path_animate(constraint="Follow Path", owner='OBJECT')

    camera.rotation_euler = (0, -0.25, 0)

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
        obj_grid.append({'obj': obj, 'original_loc': original_loc, 'translated_loc': translated_loc})

    obj_grid = np.array(obj_grid).reshape(gol.grid.shape)

    for frame in range(0, NUM_FRAMES+1):
        if frame % 10 == 0:
            print("Updating frame {}".format(frame))

        bpy.context.scene.frame_set(frame)
        # When reaching final frame, clear handlers
        if (frame % NUM_FRAMES_CHANGE) == 0:
            if frame < MIRROR_FRAME:
                gol_grid_cache.append(gol.grid)
            elif frame > MIRROR_FRAME:
                gol.grid = gol_grid_cache.pop()
        automata_blender_utils.update_grid(obj_grid, gol, obj_updater)
        if frame >= NUM_FRAMES:
            bpy.app.handlers.frame_change_pre.clear()
            bpy.context.scene.frame_set(0)

    print("-----------------------")


main()
