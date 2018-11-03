import bpy
import bmesh
from mathutils import Vector
import numpy as np

# Blender import system clutter
import sys
from pathlib import Path

SRC_PATH = Path.home() / "python_workspace/data-science-learning/graphics/spirograph"
UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
sys.path.append(str(SRC_PATH))
sys.path.append(str(UTILS_PATH))

import Spirograph
import utils.blender_utils
import importlib
importlib.reload(Spirograph)
importlib.reload(utils.blender_utils)
from Spirograph import Spirograph
from utils.blender_utils import delete_all, create_line, create_circle


def add_to_scene(spirograph: Spirograph, obj=None):
    # create and move circles
    outer_circle = create_circle(spirograph.R, segments=64, name='outer_circle')
    outer_circle.location = spirograph.origin
    inner_circle = create_circle(spirograph.r, segments=64, name='inner_circle')
    inner_circle_loc = spirograph.get_inner_circle_center()
    inner_circle.location = inner_circle_loc

    # create and move hypotrochoid
    hypotrochoid = create_line((0, 0, 0),
                               (0+spirograph.d, 0, 0), name='line')
    hypotrochoid.location = inner_circle_loc

    # create vertex group
    vg = hypotrochoid.vertex_groups.new(name="emitter")
    vg.add([1], 1.0, 'ADD')

    hypotrochoid.modifiers.new("part", type='PARTICLE_SYSTEM')
    part = hypotrochoid.particle_systems[0]
    part.settings.emit_from = 'VERT'
    part.vertex_group_density = "emitter"
    part.settings.timestep = 0.0001
    part.settings.lifetime = NUM_FRAMES
    part.settings.particle_size = 1.0
    #part.settings.render_type = 'LINE'

    if obj:
        part.settings.render_type = 'OBJECT'
        part.settings.dupli_object = obj

    return outer_circle, inner_circle, hypotrochoid


if __name__ == "__main__":
    NUM_FRAMES = 150
    NUM_FRAMES_CHANGE = 1
    bpy.context.scene.frame_end = NUM_FRAMES

    #bpy.data.objects['Camera'].location = (0, 0, 30)
    #bpy.data.objects['Camera'].rotation_euler = (0, 0, 0)

    delete_all()
    particles_obj = create_circle(1, segments=1, name='copy_particle')

    R = 5
    dist = R*3
    cols = 10
    rows = 10
    spirographs = []

    for col in range(cols):
        for row in range(rows):
            origin = (dist*col, dist*row, 0)
            spirograph = Spirograph(origin=origin, R=R, r=row/2+0.1, d=row/2+0.1, angle=0, theta=col/10+0.1)

            o_c, i_c, hyp = add_to_scene(spirograph, obj=None)
            spirograph.b_outer_circle = o_c
            spirograph.b_inner_circle = i_c
            spirograph.b_hypotrochoid = hyp


            spirographs.append(spirograph)

    hypotrochoid_rot_theta = np.array([0., 0., 0.4])
    #hypotrochoid.animation_data_clear()
    for frame in range(1, NUM_FRAMES+1):
        if frame % 10 == 0:
            print("Updating frame {}".format(frame))
        bpy.context.scene.frame_set(frame)
        # When reaching final frame, clear handlers
        if frame >= NUM_FRAMES:
            bpy.app.handlers.frame_change_pre.clear()
            bpy.context.scene.frame_set(1)
        elif (frame % NUM_FRAMES_CHANGE) == 0:
            for spirograph in spirographs:
                inner_circle = spirograph.b_inner_circle
                hypotrochoid = spirograph.b_hypotrochoid
                spirograph.update()
                # move inner circle
                inner_circle_loc = spirograph.get_inner_circle_center()
                inner_circle.location = inner_circle_loc
                inner_circle.keyframe_insert("location")
                # move hypotrochoid
                hypotrochoid.location = inner_circle_loc
                hypotrochoid.keyframe_insert("location")
                # rotate hypotrochoid
                curr_h_angle = spirograph.get_hypotrochoid_angle()
                hypotrochoid.rotation_euler = tuple(np.array(hypotrochoid.rotation_euler) +
                                                    hypotrochoid_rot_theta)
                hypotrochoid.keyframe_insert("rotation_euler")