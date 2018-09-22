import bpy


def delete_all(obj_type: str='MESH'):
    """Delete all objects of the given type from the current scene"""

    for obj in bpy.data.objects:
        obj.hide = False
        obj.select = obj.type == obj_type

    bpy.ops.object.delete(use_global=True)
