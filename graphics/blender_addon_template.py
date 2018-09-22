# Adapted from the book "The Blender Python API: Precision 3D Modeling and Add-on Development"
# by Chris Conlan

bl_info = {
    "name": "Add-on Template",
    "author": "Alex Martinelli",
    "location": "View3D > Tools > Simple Addon",
    "version": (1, 0, 0),
    "blender": (2, 7, 8),
    "description": "Template",
    "category": "Development"
}

import bpy


# Panel takes care of the UI components
class SimplePanel(bpy.types.Panel):
    # Hierarchically define location of the add-on in the Blender UI
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
    bl_context = "objectmode"
    bl_category = "Test Add-On"
    bl_label = "Template"

    # define panel UI components
    def draw(self, context):
        # sample button
        self.layout.operator("object.simple_operator",
                             text="Template operator")
        # sample int value
        self.layout.prop(context.scene, 'my_int_prop')

    @classmethod
    def register(cls):
        print("Registered class: %s " % cls.bl_label)
        # Register properties related to the class here.


    @classmethod
    def unregister(cls):
        print("Unregistered class: %s " % cls.bl_label)
        # Delete parameters related to the class here


# Operator is the actual logic behind the add-of
class SimpleOperator(bpy.types.Operator):
    bl_idname = "object.simple_operator"
    bl_label = "Template"

    def execute(self, context):
        # example of adding a monkey to the scene
        bpy.ops.mesh.primitive_monkey_add(
            radius=context.scene.my_int_prop,
            location=(0, 0, 0))
        # better to return this string when done with the execution work
        return {'FINISHED'}

    @classmethod
    def register(cls):
        print("Registered operator: %s " % cls.bl_label)
        # Register properties related to the class here
        bpy.types.Scene.my_int_prop = bpy.props.IntProperty(name="My Int",
                                                            description="Sample integer property to print to user",
                                                            default=123,
                                                            min=100,
                                                            max=200)

    @classmethod
    def unregister(cls):
        print("Unregistered operator: %s " % cls.bl_label)
        # Delete parameters related to the class here


def register():

    # Implicitly register objects inheriting bpy.types in current file and scope
    #bpy.utils.register_module(__name__)

    # Or explicitly register objects
    bpy.utils.register_class(SimpleOperator)
    bpy.utils.register_class(SimplePanel)

    print("%s registration complete\n" % bl_info.get('name'))


def unregister():
    # Always unregister in reverse order to prevent error due to
    # interdependencies

    # Explicitly unregister objects
    bpy.utils.unregister_class(SimpleOperator)
    bpy.utils.unregister_class(SimplePanel)

    # Or unregister objects inheriting bpy.types in current file and scope
    #bpy.utils.unregister_module(__name__)

    print("%s unregister complete\n" % bl_info.get('name'))


# Called only when running the script from Blender
# when distributed as plugin register() and unregister() are used
if __name__ == "__main__":

    try:
        unregister()
    except Exception as e:
        print(e)
        pass

    register()
