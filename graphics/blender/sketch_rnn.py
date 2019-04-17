# Blender import system clutter
import bpy
import bmesh
from mathutils import Vector
import numpy as np
import json

import sys
from pathlib import Path

UTILS_PATH = Path.home() / "Documents/python_workspace/data-science-learning"
SRC_PATH = UTILS_PATH / "graphics/blender"
sys.path.append(str(UTILS_PATH))
sys.path.append(str(SRC_PATH))

import importlib
import ds_utils.blender_utils
importlib.reload(ds_utils.blender_utils)
from ds_utils.blender_utils import init_grease_pencil, get_grease_pencil, get_grease_pencil_layer


def load_quickdraw_file(filepath: str, max_drawings=100):
    drawings = []
    with open(filepath) as f:
        drawings_count = 0
        for line in f:
            drawing = json.loads(line)
            drawings.append(drawing['drawing'])
            drawings_count += 1
            if drawings_count >= max_drawings:
                break
    return drawings


def draw_quickdraw_entry(gp_frame, entry, translate_vector=(0, 0, 0)):
    for stroke in entry:
        gp_stroke = gp_frame.strokes.new()
        points = list(zip(stroke[0], stroke[1]))
        gp_stroke.points.add(count=len(points))
        for i, point in enumerate(points):
            gp_stroke.points[i].co = np.array((point[0], point[1], 0)) + translate_vector


def draw_stroke(gp_frame, stroke, scale=1., translate_vector=(0, 0, 0)):
    gp_stroke = gp_frame.strokes.new()
    points = list(zip(stroke[0], stroke[1]))
    gp_stroke.points.add(count=len(points))
    for i, point in enumerate(points):
        new_point = (np.array((point[0], point[1], 0))*scale) + np.array(translate_vector)
        gp_stroke.points[i].co = (new_point[0], new_point[2], -new_point[1])


def quickdraw_main():
    NB_ROWS = 50
    NB_COLS = 20
    CELL_SPACING = 8

    NUM_FRAMES = 50
    FRAMES_SPACING = 1
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = NUM_FRAMES * FRAMES_SPACING

    MAX_DRAWINGS = 50

    print("Loading dataset")
    data_dir = Path.home() / "Documents/datasets/quickdraw/simplified"
    filepaths = list(data_dir.glob('*.ndjson'))[:(NB_ROWS*NB_COLS)]
    quickdraw_entries = [load_quickdraw_file(filepath=filepath, max_drawings=MAX_DRAWINGS) for filepath in filepaths]
    print("Dataset Loaded")

    gpencil = get_grease_pencil()
    gp_layers = [get_grease_pencil_layer(gpencil, "layer_{}".format(i), True) for i in range(len(filepaths))]

    z = 0
    for frame in range(NUM_FRAMES):
        if frame % 100 == 0:
            print("Updating frame {}".format(frame))
        for row in range(NB_ROWS):
            for col in range(NB_COLS):
                gp_layer = gp_layers[row * NB_COLS + col]
                drawings = quickdraw_entries[row * NB_COLS + col]

                # Suboptimal way
                drawing_idx = 0
                total_strokes = len(drawings[0])
                while frame >= total_strokes and drawing_idx<len(drawings)-1:
                    drawing_idx += 1
                    total_strokes += len(drawings[drawing_idx])

                stroke_idx = frame-(total_strokes-len(drawings[drawing_idx]))
                drawing = drawings[drawing_idx]
                if stroke_idx >= len(drawing):
                    pass
                else:
                    if frame == (total_strokes-len(drawings[drawing_idx])) or frame==0:
                        gp_frame = gp_layer.frames.new(frame)
                    else:
                        gp_frame = gp_layer.frames.copy(gp_layer.frames[frame-1])

                    draw_stroke(gp_frame, drawing[stroke_idx], 0.02,
                                (row*CELL_SPACING, col*CELL_SPACING, col*CELL_SPACING))


def sketch_rnn_draw():
    with open("/tmp/stroke.npy", 'rb') as f:
        stroke = np.load(f)[0]

    gp_layer = init_grease_pencil()
    gp_frame = gp_layer.frames.new(0)
    gp_stroke = gp_frame.strokes.new()
    gp_stroke.points.add(count=len(stroke))
    for i, point in enumerate(stroke):
        gp_stroke.points[i].co = point


quickdraw_main()