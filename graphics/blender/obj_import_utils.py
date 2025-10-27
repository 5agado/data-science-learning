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


def main(input_dir: Path, collection_name: str, file_format='obj', max_num_objs=100, src_object_name: str="",
         obj_scale=1.0, translation_scale=0, decimate_ratio=1.0, hide_nb_frames=0):
    """
    Import all objects from the given input directory.
    :param input_dir:
    :param collection_name: add imported objects to this collection (create one if it doesn't exist)
    :param file_format: obj or ply
    :param max_num_objs:
    :param src_object_name: name of the object to use as template for the imported ones
    :param obj_scale:
    :param translation_scale:
    :param decimate_ratio:
    :param hide_nb_frames: distance in frames between each object animation change
    """
    # collection setup, objects will be added to this collection
    if collection_name not in bpy.data.collections:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection_name]


    # run animation
    filenames = list(input_dir.glob(f'*/*.{file_format}'))[:max_num_objs]
    for i, filename in enumerate(filenames):
        if file_format == 'ply':
            imported_object = bpy.ops.import_mesh.ply(filepath=str(filename))
        elif file_format == 'obj':
            imported_object = bpy.ops.import_scene.obj(filepath=str(filename))
        elif file_format == 'glb':
            imported_object = bpy.ops.import_scene.gltf(filepath=str(filename), files=[{"name":"mesh.glb"}], loglevel=20)
        else:
            print(f'No such file format {file_format}')
            sys.exit()

        # get imported object and set context in order to apply transformation and modifiers
        #obj = bpy.context.selected_objects[-1]
        #bpy.context.view_layer.objects.active = obj
        obj = bpy.context.view_layer.objects.active
        for child in obj.children_recursive:
            for col in list(obj.users_collection): # Remove from other collections
                col.objects.unlink(child)
            bpy.context.collection.objects.link(child)
        for col in list(obj.users_collection): # Remove from other collections
            col.objects.unlink(obj)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj = list(obj.children_recursive)[-1]
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # scale
        if obj_scale != 1.0:
            bpy.ops.transform.resize(value=tuple([obj_scale] * 3), constraint_axis=(False, False, False))
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # translate
        if translation_scale != 0:
            obj_pos = (i*translation_scale, 0, 0)
            bpy.ops.transform.translate(value=obj_pos)
            bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

        # set name and shade-smooth
        obj.name = f'obj_{i:03}'
        bpy.ops.object.shade_smooth()

        # apply modifiers
        if decimate_ratio != 1.0:
            modifier = obj.modifiers.new('DecimateMod', 'DECIMATE')
            modifier.ratio = decimate_ratio
            bpy.ops.object.modifier_apply(modifier="DecimateMod")

        # hide object
        if hide_nb_frames != 0:
            hide_obj(obj, 0, True)
            hide_obj(obj, i * hide_nb_frames, False)
            hide_obj(obj, (i + 1) * hide_nb_frames, True)

        # material
        if src_object_name:
            src_obj = bpy.data.objects[src_object_name]
            obj.active_material = src_obj.active_material.copy()
            obj_imgpath = str(filename).replace('obj', 'png') # assumes texture image is in the same location of obj, with same name
            image = bpy.data.images.load(filepath = obj_imgpath)
            obj.material_slots[0].material.node_tree.nodes["color_img"].image = image
            obj.material_slots[0].material.node_tree.nodes["normal_img"].image = image

        obj.select_set(state=False)

        print(f'({i}/{len(filenames)}) Imported object: ', obj.name)

main(input_dir=Path(r'C:\Users'),
     collection_name='Aircrafts', file_format='glb',max_num_objs=100,
     src_object_name='')