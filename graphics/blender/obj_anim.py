from pathlib import Path
import bpy
import sys


def hide(obj, frame, val):
    bpy.context.scene.frame_set(frame)
    obj.hide_viewport = val
    obj.hide_render = obj.hide_viewport
    obj.keyframe_insert("hide_viewport")
    obj.keyframe_insert("hide_render")


files_loc = Path('D:\\generated_data\\pifuhd\\res_women\\recon')
files_format = 'obj' # obj ply
obj_scale = 1.
translation_scale = 0
nb_frames = 24 # distance in frames between each object animation change
src_obj = bpy.context.selected_objects[0]

# collection setup
collection_name = 'women'
if collection_name not in bpy.data.collections:
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection_name]


# run animation
for i, filename in enumerate(list(files_loc.glob(f'*.{files_format}'))[:100]):
    if files_format == 'ply':
        imported_object = bpy.ops.import_mesh.ply(filepath=str(filename))
    elif files_format == 'obj':
        imported_object = bpy.ops.import_scene.obj(filepath=str(filename))
    else:
        print(f'No such file format {files_format}')
        sys.exit()

    obj = bpy.context.selected_objects[0]  ####<--Fix

    # scale
    bpy.ops.transform.resize(value=tuple([obj_scale] * 3), constraint_axis=(False, False, False))

    # translate
    # obj_pos = (i*translation_scale, 0, 0)
    # bpy.ops.transform.translate(value=obj_pos)

    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

    # assignment
    # obj = bpy.context.active_object
    obj.name = f'obj_dress_{i:03}'
    bpy.ops.object.shade_smooth()
    
    # apply modifiers
    modifier = obj.modifiers.new('DecimateMod', 'DECIMATE')
    modifier.ratio = 0.1
    bpy.context.scene.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="DecimateMod")


    # hide
    hide(obj, 0, True)
    hide(obj, i * nb_frames, False)
    hide(obj, (i + 1) * nb_frames, True)
    # bpy.context.collection.objects.link(obj)

    # material
    
    obj.active_material = src_obj.active_material.copy()
    obj_imgpath = str(filename).replace('obj', 'png')
    image = bpy.data.images.load(filepath = obj_imgpath)
    obj.material_slots[0].material.node_tree.nodes["Image Texture.001"].image = image
    obj.material_slots[0].material.node_tree.nodes["Image Texture"].image = image
    
    print('Imported name: ', obj.name)