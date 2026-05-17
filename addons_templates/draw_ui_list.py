import bpy
from bl_ui.generic_ui_list import draw_ui_list


class MyPropGroup(bpy.types.PropertyGroup):
    complex_integer: bpy.props.IntVectorProperty(
        name="Complex Integer",
        size=2,
        default=(0, 0),
    )
    transform_flags: bpy.props.BoolVectorProperty(
        name="Transform Flags",
        size=2,
        default=(False, False),
    )


class MyPanel(bpy.types.Panel):
    bl_label = "My Label"
    bl_idname = "SCENE_PT_list_demo"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "My Category"

    def draw(self, context):
        layout = self.layout
        draw_ui_list(
            layout,
            context,
            list_path="scene.my_list",
            active_index_path="scene.my_list_active_index",
            unique_id="my_list_id",
        )


classes = [
    MyPropGroup,
    MyPanel,
]

class_register, class_unregister = bpy.utils.register_classes_factory(classes)


def register():
    class_register()
    bpy.types.Scene.my_list = bpy.props.CollectionProperty(type=MyPropGroup)
    bpy.types.Scene.my_list_active_index = bpy.props.IntProperty()


def unregister():
    class_unregister()
    del bpy.types.Scene.my_list
    del bpy.types.Scene.my_list_active_index

if __name__ == "__main__":
    register()