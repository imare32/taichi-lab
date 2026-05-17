import math

import bpy
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)

# =========================
# 1. Taichi 计算核心
# =========================

MAX_PINS = 1000
MAX_LINES = 10000

img_field = ti.field(ti.f32, shape=(1024, 1024))
pins_field = ti.Vector.field(2, ti.f32, shape=MAX_PINS)
lines_field = ti.field(ti.i32, shape=(MAX_LINES, 2))
current_pin = ti.field(ti.i32, shape=())
num_lines = ti.field(ti.i32, shape=())


@ti.kernel
def init_pins(n_pins: int, center: ti.math.vec2, radius: float):
    for i in range(n_pins):
        angle = 2.0 * math.pi * i / n_pins
        pins_field[i] = center + radius * ti.math.vec2(ti.cos(angle), ti.sin(angle))
    current_pin[None] = 0
    num_lines[None] = 0


@ti.func
def sample_bilinear(x: float, y: float) -> float:
    """双线性插值采样，提供平滑的亚像素读取（抗锯齿）"""
    w, h = img_field.shape
    x = ti.max(0.0, ti.min(x, w - 1.001))
    y = ti.max(0.0, ti.min(y, h - 1.001))

    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0
    wy = y - y0

    return img_field[x0, y0] * (1 - wx) * (1 - wy) + img_field[x1, y0] * wx * (1 - wy) + img_field[x0, y1] * (1 - wx) * wy + img_field[x1, y1] * wx * wy


@ti.func
def subtract_bilinear(x: float, y: float, strength: float):
    """双线性插值扣减，带线宽与抗锯齿的物理光栅化"""
    w, h = img_field.shape
    x = ti.max(0.0, ti.min(x, w - 1.001))
    y = ti.max(0.0, ti.min(y, h - 1.001))

    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0
    wy = y - y0

    img_field[x0, y0] = ti.max(0.0, img_field[x0, y0] - strength * (1 - wx) * (1 - wy))
    img_field[x1, y0] = ti.max(0.0, img_field[x1, y0] - strength * wx * (1 - wy))
    img_field[x0, y1] = ti.max(0.0, img_field[x0, y1] - strength * (1 - wx) * wy)
    img_field[x1, y1] = ti.max(0.0, img_field[x1, y1] - strength * wx * wy)


@ti.func
def get_line_score(p1_idx: int, p2_idx: int) -> float:
    p1 = pins_field[p1_idx]
    p2 = pins_field[p2_idx]

    w, h = img_field.shape
    p1_px = p1 * ti.math.vec2(w - 1, h - 1)
    p2_px = p2 * ti.math.vec2(w - 1, h - 1)

    # 【核心修复】动态步长：用像素级的欧氏距离决定采样次数
    dist = (p1_px - p2_px).norm()
    steps = ti.max(int(dist), 1)

    score = 0.0
    for i in range(steps):
        t = i / float(steps)
        pos = p1_px + t * (p2_px - p1_px)
        score += sample_bilinear(pos.x, pos.y)

    return score / float(steps)


@ti.func
def subtract_line(p1_idx: int, p2_idx: int, strength: float):
    p1 = pins_field[p1_idx]
    p2 = pins_field[p2_idx]

    w, h = img_field.shape
    p1_px = p1 * ti.math.vec2(w - 1, h - 1)
    p2_px = p2 * ti.math.vec2(w - 1, h - 1)

    dist = (p1_px - p2_px).norm()
    steps = ti.max(int(dist), 1)

    for i in range(steps):
        t = i / float(steps)
        pos = p1_px + t * (p2_px - p1_px)
        subtract_bilinear(pos.x, pos.y, strength)


@ti.kernel
def find_next_line(n_pins: int, strength: float, min_dist: int):
    p1 = current_pin[None]
    best_p2 = -1
    max_score = -1.0

    for i in range(n_pins):
        dist = abs(i - p1)
        if dist > n_pins // 2:
            dist = n_pins - dist
        if dist < min_dist:
            continue

        score = get_line_score(p1, i)
        if score > max_score:
            max_score = score
            best_p2 = i

    if best_p2 != -1:
        subtract_line(p1, best_p2, strength)
        lines_field[num_lines[None], 0] = p1
        lines_field[num_lines[None], 1] = best_p2
        num_lines[None] += 1
        current_pin[None] = best_p2


# =========================
# 2. Blender 交互逻辑
# =========================
def get_image_data(image, invert=True, gamma=1.0):
    width, height = image.size
    pixels = np.array(image.pixels[:], dtype=np.float32)
    pixels = pixels.reshape((height, width, 4))

    # 【核心修复】由于 Blender 像素是线性空间，使用 Rec. 709 权重计算亮度
    luminance = 0.2126 * pixels[:, :, 0] + 0.7152 * pixels[:, :, 1] + 0.0722 * pixels[:, :, 2]

    # 色调映射映射 (Gamma)
    if gamma != 1.0:
        luminance = np.power(luminance, gamma)

    if invert:
        grayscale = 1.0 - luminance
    else:
        grayscale = luminance

    target_res = 1024

    # 【核心修复】Numpy 双线性降采样，防止粗暴切片导致的摩尔纹
    if width != target_res or height != target_res:
        x = np.linspace(0, width - 1, target_res)
        y = np.linspace(0, height - 1, target_res)

        # 提取整数和小数部分
        x_idx = np.clip(np.floor(x).astype(int), 0, width - 2)
        y_idx = np.clip(np.floor(y).astype(int), 0, height - 2)
        x_w = x - x_idx
        y_w = y - y_idx

        # 生成网格
        X_idx, Y_idx = np.meshgrid(x_idx, y_idx)
        X_w, Y_w = np.meshgrid(x_w, y_w)

        p00 = grayscale[Y_idx, X_idx]
        p10 = grayscale[Y_idx, X_idx + 1]
        p01 = grayscale[Y_idx + 1, X_idx]
        p11 = grayscale[Y_idx + 1, X_idx + 1]

        # 双线性计算
        grayscale = p00 * (1 - X_w) * (1 - Y_w) + p10 * X_w * (1 - Y_w) + p01 * (1 - X_w) * Y_w + p11 * X_w * Y_w

    # 翻转数组以匹配 Taichi 的二维数据结构 [x, y]
    return grayscale.T


def reset_string_art(scene):
    img_name = scene.taichi_string_img
    if not img_name or img_name not in bpy.data.images:
        img_field.fill(0)
        return

    img = bpy.data.images[img_name]
    try:
        data = get_image_data(img, invert=scene.taichi_string_invert, gamma=scene.taichi_string_gamma)
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    img_field.from_numpy(data)

    init_pins(scene.taichi_string_pins, ti.math.vec2(0.5, 0.5), 0.45)

    obj_name = scene.get("taichi_string_obj", "")
    if obj_name and obj_name in bpy.data.objects:
        obj = bpy.data.objects[obj_name]
        mesh = obj.data
        mesh.clear_geometry()
        mesh.update()


def update_string_art(scene):
    if scene.frame_current <= scene.frame_start:
        reset_string_art(scene)
        return

    obj_name = scene.get("taichi_string_obj", "")
    if not obj_name or obj_name not in bpy.data.objects:
        return

    obj = bpy.data.objects[obj_name]
    mesh = obj.data

    n_pins = scene.taichi_string_pins
    max_lines_total = scene.taichi_string_max_lines
    lines_per_frame = scene.taichi_string_speed
    strength = scene.taichi_string_strength
    min_dist = scene.taichi_string_min_dist

    if num_lines[None] >= max_lines_total:
        return

    for _ in range(lines_per_frame):
        if num_lines[None] < max_lines_total:
            find_next_line(n_pins, strength, min_dist)

    total_l = num_lines[None]
    if total_l == 0:
        return

    pins_np = pins_field.to_numpy()[:n_pins]
    verts = (pins_np - 0.5) * 10.0
    verts_3d = np.zeros((n_pins, 3), dtype=np.float32)
    verts_3d[:, 0] = verts[:, 0]
    verts_3d[:, 1] = verts[:, 1]

    edges = lines_field.to_numpy()[:total_l]

    if len(mesh.edges) != total_l:
        mesh.clear_geometry()
        mesh.from_pydata(verts_3d.tolist(), edges.tolist(), [])
        mesh.update()


@bpy.app.handlers.persistent
def string_art_handler(scene):
    obj_name = scene.get("taichi_string_obj", "")
    if obj_name in bpy.data.objects:
        update_string_art(scene)


# =========================
# 3. UI 操作符与面板
# =========================


class TAI_OT_create_string_art(bpy.types.Operator):
    bl_idname = "taichi.create_string_art"
    bl_label = "Initialize String Art"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        mesh = bpy.data.meshes.new("StringArt_Mesh")
        obj = bpy.data.objects.new("StringArt_Obj", mesh)
        context.collection.objects.link(obj)
        scene["taichi_string_obj"] = obj.name
        reset_string_art(scene)
        return {"FINISHED"}


class TAI_PT_string_art_panel(bpy.types.Panel):
    bl_label = "Taichi String Art"
    bl_idname = "TAI_PT_string_art_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop_search(scene, "taichi_string_img", bpy.data, "images", text="Image")
        layout.prop(scene, "taichi_string_invert")
        layout.prop(scene, "taichi_string_gamma")
        layout.prop(scene, "taichi_string_pins")
        layout.prop(scene, "taichi_string_max_lines")
        layout.prop(scene, "taichi_string_speed")
        layout.prop(scene, "taichi_string_strength")
        layout.prop(scene, "taichi_string_min_dist")

        layout.operator(TAI_OT_create_string_art.bl_idname, icon="IMAGE_DATA")

        obj_name = scene.get("taichi_string_obj", "")
        if obj_name in bpy.data.objects:
            layout.label(text=f"Lines: {num_lines[None]} / {scene.taichi_string_max_lines}")


def register():
    bpy.utils.register_class(TAI_OT_create_string_art)
    bpy.utils.register_class(TAI_PT_string_art_panel)

    bpy.types.Scene.taichi_string_img = bpy.props.StringProperty(name="Image")
    bpy.types.Scene.taichi_string_invert = bpy.props.BoolProperty(name="Invert Intensity", default=True, description="Target dark areas (True) or bright areas (False)")
    bpy.types.Scene.taichi_string_gamma = bpy.props.FloatProperty(name="Contrast (Gamma)", default=1.0, min=0.1, max=5.0, description="Higher values increase contrast of dark areas")
    bpy.types.Scene.taichi_string_pins = bpy.props.IntProperty(name="Pins", default=200, min=10, max=MAX_PINS)
    bpy.types.Scene.taichi_string_max_lines = bpy.props.IntProperty(name="Max Lines", default=3000, min=100, max=MAX_LINES)
    bpy.types.Scene.taichi_string_speed = bpy.props.IntProperty(name="Lines Per Frame", default=10, min=1, max=100)
    bpy.types.Scene.taichi_string_strength = bpy.props.FloatProperty(name="Strength", default=0.1, min=0.001, max=1.0)
    bpy.types.Scene.taichi_string_min_dist = bpy.props.IntProperty(name="Min Pin Dist", default=20, min=1, max=100)

    if string_art_handler not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(string_art_handler)


def unregister():
    bpy.utils.unregister_class(TAI_OT_create_string_art)
    bpy.utils.unregister_class(TAI_PT_string_art_panel)

    del bpy.types.Scene.taichi_string_img
    del bpy.types.Scene.taichi_string_invert
    del bpy.types.Scene.taichi_string_gamma
    del bpy.types.Scene.taichi_string_pins
    del bpy.types.Scene.taichi_string_max_lines
    del bpy.types.Scene.taichi_string_speed
    del bpy.types.Scene.taichi_string_strength
    del bpy.types.Scene.taichi_string_min_dist

    if string_art_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(string_art_handler)


"""
后续处理几何节点:

    Mesh to curve -> Set Curve Radius 0.004 -> Curves to grease pencil -> Set grease pencil color (Opacity 0.5)
"""

if __name__ == "__main__" or __name__ == "<run_path>":
    register()
