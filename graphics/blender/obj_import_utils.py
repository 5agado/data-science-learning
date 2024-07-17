"""
Set of utils to programmatically import objects in Blender, apply modifiers and materials.
"""

from pathlib import Path
import bpy
import sys


def hide_obj(obj, frame, val):
    bpy.context.scene.frame_set(frame)
    obj.hide_viewport = val
    obj.hide_render = obj.hide_viewport
    obj.keyframe_insert("hide_viewport")
    obj.keyframe_insert("hide_render")


def main(input_dir: Path, collection_name: str, src_object_name: str, file_format='obj'):
    """
    Import all objects from the given input directory.
    :param input_dir:
    :param collection_name: add imported objects to this collection (create one if it doesn't exist)
    :param src_object_name: name of the object to use as template for the imported ones
    :param file_format: obj or ply
    :return:
    """
    max_num_objs = 100
    obj_scale = 1.
    translation_scale = 0
    nb_frames = 24 # distance in frames between each object animation change
    src_obj = bpy.data.objects[src_object_name]

    # collection setup
    # add objects to this collection
    if collection_name not in bpy.data.collections:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection_name]


    # run animation
    filenames = list(input_dir.glob(f'*.{file_format}'))[:max_num_objs]
    for i, filename in enumerate(filenames):
        if file_format == 'ply':
            imported_object = bpy.ops.import_mesh.ply(filepath=str(filename))
        elif file_format == 'obj':
            imported_object = bpy.ops.import_scene.obj(filepath=str(filename))
        else:
            print(f'No such file format {file_format}')
            sys.exit()

        # get imported object and set context in order to apply transformation and modifiers
        obj = bpy.context.selected_objects[-1]
        bpy.context.view_layer.objects.active = obj

        # scale
        bpy.ops.transform.resize(value=tuple([obj_scale] * 3), constraint_axis=(False, False, False))

        # translate
        # obj_pos = (i*translation_scale, 0, 0)
        # bpy.ops.transform.translate(value=obj_pos)

        bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

        # set name and shade-smooth
        obj.name = f'obj_{i:03}'
        bpy.ops.object.shade_smooth()

        # apply modifiers
        modifier = obj.modifiers.new('DecimateMod', 'DECIMATE')
        modifier.ratio = 0.1
        bpy.ops.object.modifier_apply(modifier="DecimateMod")

        # hide object
        # hide_obj(obj, 0, True)
        # hide_obj(obj, i * nb_frames, False)
        # hide_obj(obj, (i + 1) * nb_frames, True)
        # bpy.context.collection.objects.link(obj)

        # material
        obj.active_material = src_obj.active_material.copy()
        obj_imgpath = str(filename).replace('obj', 'png') # assumes texture image is in the same location of obj, with same name
        image = bpy.data.images.load(filepath = obj_imgpath)
        obj.material_slots[0].material.node_tree.nodes["color_img"].image = image
        obj.material_slots[0].material.node_tree.nodes["normal_img"].image = image

        obj.select_set(state=False)

        print(f'({i}/{len(filenames)}) Imported object: ', obj.name)

main(input_dir=Path('D:\\generated_data\\pifuhd\\astronaut'), collection_name='collection',
     src_object_name='result_0000_256')