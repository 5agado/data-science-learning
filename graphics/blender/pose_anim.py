# Blender import system clutter
import bpy
import bmesh
from mathutils import Vector
import numpy as np
import json

import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
sys.path.append(str(UTILS_PATH))

import importlib
import ds_utils.blender_utils
importlib.reload(ds_utils.blender_utils)
from ds_utils.blender_utils import init_grease_pencil, delete_all, draw_segment, create_object


def _update_obj(obj_name: str, vertices):
    obj = bpy.context.scene.objects[obj_name]
    mesh = obj.data
    bm = bmesh.new()

    # convert the current mesh to a bmesh (must be in edit mode)
    bpy.ops.object.mode_set(mode='EDIT')
    bm.from_mesh(mesh)
    bpy.ops.object.mode_set(mode='OBJECT')  # return to object mode

    for v in bm.verts:
        bm.verts.remove(v)
    for v in vertices:
        bm.verts.new(v)

    # make the bmesh the object's mesh
    bm.to_mesh(mesh)
    bm.free()  # always do this when finished


def _init_shape_keys(obj):
    sk_basis = obj.shape_key_add(name='Basis')
    sk_basis.interpolation = 'KEY_LINEAR'
    obj.data.shape_keys.use_relative = False


def _update_shape_keys(obj, vertices, frame, frames_spacing=1):
    # Create new shape-key block
    block = obj.shape_key_add(name=str(frame), from_mix=False)  # returns a key_blocks member
    block.interpolation = 'KEY_LINEAR'
    block.value = 0

    # Update vertices position
    for (vert, co) in zip(block.data, vertices):
        if np.all(co == 0.):
            continue
        else:
            vert.co = co

    # Keyframe evaluation time
    obj.data.shape_keys.eval_time = frame * 10
    obj.data.shape_keys.keyframe_insert(data_path='eval_time', frame=frame*frames_spacing)


def main(points_filepath: str, scale_factor=10, translate_vector=(0, 0, 0), frames_spacing=1):
    # load pose points
    pose_points = np.load(points_filepath)

    # scale
    pose_points /= scale_factor

    # TODO need like this because number of entries is variable
    # if necessary expand pose coordinates to 3D and translate
    for i in range(len(pose_points)):
        if pose_points[i].shape[-1] == 2:
            z = np.zeros(pose_points.shape[:-1] + (1,), dtype=pose_points.dtype)
            pose_points = np.concatenate((pose_points, z), axis=-1)

        # translate
        pose_points += translate_vector

    NUM_FRAMES = len(pose_points)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = NUM_FRAMES * frames_spacing

    #delete_all()

    # Create initial object (or update vertices if one already exist
    obj_name = 'pose'
    if obj_name not in bpy.context.scene.objects:
        create_object(pose_points[0], edges=[], faces=[], obj_name=obj_name)
    else:
        _update_obj(obj_name, pose_points[0])

    obj = bpy.context.scene.objects[obj_name]
    _init_shape_keys(obj)

    # Run animation across frames
    #gp_layer = init_grease_pencil(clear_layer=True)
    for frame in range(1, NUM_FRAMES):  # start from 1 as first frame points have been used to instantiate object
        if frame % 100 == 0:
            print("Updating frame {}".format(frame))

        # Shape keys
        _update_shape_keys(obj, pose_points[frame], frame, frames_spacing)

        # GP
        #gp_frame = gp_layer.frames.new(frame * frames_spacing)
        #for segment in pose_points[frame]:
        #    draw_segment(gp_frame, segment)


main(points_filepath=str(Path.home() / "all_keypoints.npy"),
     frames_spacing=10)
