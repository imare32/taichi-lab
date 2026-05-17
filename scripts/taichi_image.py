import bpy
import numpy as np
import taichi as ti

bl_info = {
    "name": "Taichi Image Generator",
    "author": "Gemini Assistant",
    "version": (1, 1),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Taichi Tab",
    "description": "Generate Taichi Logo with adjustable eye radius",
    "category": "Image",
}

# --- Taichi Setup ---
if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu, log_level=ti.INFO)
    except:
        ti.init(arch=ti.cpu)

# Constants
n = 512
pixels = ti.field(dtype=ti.f32, shape=(n, n))


@ti.func
def taichi_logo(pos, scale: float = 1 / 1.11, angle: float = 0.0, eye_radius: float = 0.08):
    # 旋转与缩放逻辑
    p_centered = pos - 0.5
    c, s = ti.cos(angle), ti.sin(angle)
    x = p_centered[0] * c - p_centered[1] * s
    y = p_centered[0] * s + p_centered[1] * c
    p = ti.Vector([x, y]) / scale + 0.5

    # 形状定义
    ret = -1
    dist_sq = (p - 0.50).norm_sqr()
    if not dist_sq <= 0.52**2:
        ret = 0
    elif not dist_sq <= 0.495**2:
        ret = 1
    elif (p - ti.Vector([0.50, 0.25])).norm_sqr() <= eye_radius**2:
        ret = 1
    elif (p - ti.Vector([0.50, 0.75])).norm_sqr() <= eye_radius**2:
        ret = 0
    elif (p - ti.Vector([0.50, 0.25])).norm_sqr() <= 0.25**2:
        ret = 0
    elif (p - ti.Vector([0.50, 0.75])).norm_sqr() <= 0.25**2:
        ret = 1
    elif p[0] < 0.5:
        ret = 1
    else:
        ret = 0
    return 1.0 - float(ret)


@ti.kernel
def paint(angle: float, eye_radius: float):
    for i, j in pixels:
        acc = 0.0
        for u, v in ti.ndrange(4, 4):
            offset_x = (u + 0.5) / 4.0
            offset_y = (v + 0.5) / 4.0
            sub_pos = ti.Vector([(i + offset_x) / n, (j + offset_y) / n])
            acc += taichi_logo(sub_pos, angle=angle, eye_radius=eye_radius)
        pixels[i, j] = acc / 16.0


def update_blender_image(angle, eye_radius):
    paint(angle, eye_radius)
    data = pixels.to_numpy()
    rgba = np.zeros((n, n, 4), dtype=np.float32)
    rgba[:, :, 0] = data
    rgba[:, :, 1] = data
    rgba[:, :, 2] = data
    rgba[:, :, 3] = 1.0
    
    pixels_flat = rgba.ravel()
    img = bpy.data.images.get("Taichi_Logo")
    if img is None:
        img = bpy.data.images.new("Taichi_Logo", width=n, height=n)
    else:
        if img.size[0] != n or img.size[1] != n:
            img.scale(n, n)
            
    img.pixels.foreach_set(pixels_flat)
    img.update()


# --- UI & Operators ---

class TAICHI_OT_render_image(bpy.types.Operator):
    bl_idname = "taichi.render_image"
    bl_label = "Render Taichi Image"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.taichi_image_props
        update_blender_image(props.angle, props.eye_radius)
        
        img = bpy.data.images.get("Taichi_Logo")
        if img:
            for area in context.screen.areas:
                if area.type == 'IMAGE_EDITOR':
                    area.spaces.active.image = img
        
        return {'FINISHED'}


class TaichiImageProperties(bpy.types.PropertyGroup):
    angle: bpy.props.FloatProperty(
        name="Rotation Angle",
        description="Angle in radians",
        default=0.0,
        update=lambda self, context: update_blender_image(self.angle, self.eye_radius)
    )
    eye_radius: bpy.props.FloatProperty(
        name="Eye Radius",
        description="Radius of the Taichi fish eyes",
        default=0.08,
        min=0.0,
        max=0.25,
        step=0.01,
        update=lambda self, context: update_blender_image(self.angle, self.eye_radius)
    )


class VIEW3D_PT_taichi_image_panel(bpy.types.Panel):
    bl_label = "Taichi Image Generator"
    bl_idname = "VIEW3D_PT_taichi_image"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout = self.layout
        props = context.scene.taichi_image_props
        
        layout.prop(props, "angle")
        layout.prop(props, "eye_radius")
        layout.operator("taichi.render_image", icon='RENDER_STILL')


classes = (
    TaichiImageProperties,
    TAICHI_OT_render_image,
    VIEW3D_PT_taichi_image_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.taichi_image_props = bpy.props.PointerProperty(type=TaichiImageProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.taichi_image_props


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
