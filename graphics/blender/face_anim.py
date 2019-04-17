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


def draw_face(gp_frame, face):
    # Jawline
    draw_segment(gp_frame, face[0:17])

    # Right Brow
    draw_segment(gp_frame, face[17:22])

    # Left Brow
    draw_segment(gp_frame, face[22:27])

    # Nose Line
    draw_segment(gp_frame, face[27:31])

    # Nose Base
    draw_segment(gp_frame, face[31:36])

    # Right Eye
    draw_segment(gp_frame, face[36:42], draw_cyclic=True)

    # Left Eye
    draw_segment(gp_frame, face[42:48], draw_cyclic=True)

    # Outside Mouth
    draw_segment(gp_frame, face[48:60], draw_cyclic=True)

    # Inside Mouth
    draw_segment(gp_frame, face[60:68], draw_cyclic=True)


def create_face(name: str, face):
    verts = face

    edges = [(i, i+1) for i in range(0, 16)]         # Jawline
    edges.extend([(i, i+1) for i in range(17, 21)])  # Right Brow
    edges.extend([(i, i+1) for i in range(22, 26)])  # Left Brow
    edges.extend([(i, i+1) for i in range(27, 30)])  # Nose Line
    edges.extend([(i, i+1) for i in range(31, 35)])  # Nose Base
    edges.extend([(i, i+1) for i in range(36, 41)])  # Right Eye
    edges.extend([(41, 36)])
    edges.extend([(i, i+1) for i in range(42, 47)])  # Left Eye
    edges.extend([(47, 42)])
    edges.extend([(i, i+1) for i in range(48, 59)])  # Outside Mouth
    edges.extend([(59, 48)])
    edges.extend([(i, i+1) for i in range(60, 67)])  # Inside Mouth
    edges.extend([(67, 60)])

    obj = create_object(verts, edges, faces=[], obj_name=name)

    return obj


def main():
    # load landmarks
    face_points = np.load(str(Path.home() / "Downloads/celeba_3d_landmarks.npy")).astype(np.float32)

    # if necessary expand face coordinates to 3D
    if face_points.shape[2] == 2:
        z = np.zeros((face_points.shape[0], face_points.shape[1], 1), dtype=face_points.dtype)
        face_points = np.concatenate((face_points, z), axis=-1)

    print(face_points.shape)

    # translate and scale
    scale_factor = 10
    translate_vector = (0, 0, 0)
    face_points /= scale_factor
    face_points += translate_vector

    NUM_FRAMES = len(face_points)
    FRAMES_SPACING = 20
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = NUM_FRAMES * FRAMES_SPACING

    delete_all()
    #gp_layer = init_grease_pencil(clear_layer=True)

    #bpy.ops.mesh.primitive_circle_add(vertices=face_points.shape[1]) #, fill_type='NGON')
    #obj = bpy.context.active_object

    #obj = bpy.context.scene.objects['3d_face']

    obj = create_face('face', face_points[0])

    sk_basis = obj.shape_key_add(name='Basis')
    sk_basis.interpolation = 'KEY_LINEAR'
    obj.data.shape_keys.use_relative = False

    for frame in range(NUM_FRAMES):
        if frame % 100 == 0:
            print("Updating frame {}".format(frame))

        # Create new shape-key block
        block = obj.shape_key_add(name=str(frame), from_mix=False)  # returns a key_blocks member
        block.interpolation = 'KEY_LINEAR'
        block.value = 0

        # Update vertices position
        for (vert, co) in zip(block.data, face_points[frame]):
            vert.co = co

        # Keyframe evaluation time
        obj.data.shape_keys.eval_time = frame * 10
        obj.data.shape_keys.keyframe_insert(data_path='eval_time', frame=frame*FRAMES_SPACING)

        # GP
        #gp_frame = gp_layer.frames.new(frame*FRAMES_SPACING)
        #draw_face(gp_frame, face_points[frame])


main()
