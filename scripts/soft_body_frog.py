import bpy
import numpy as np
import taichi as ti
import math

# --- Taichi Setup ---
if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu, log_level=ti.INFO)
    except:
        ti.init(arch=ti.cpu)

# --- Constants ---
NUM_BLOB_POINTS = 16
TOTAL_POINTS = NUM_BLOB_POINTS

# --- Taichi Fields ---
pos = ti.Vector.field(2, dtype=ti.f32, shape=TOTAL_POINTS)
ppos = ti.Vector.field(2, dtype=ti.f32, shape=TOTAL_POINTS)
disp = ti.Vector.field(2, dtype=ti.f32, shape=TOTAL_POINTS)
disp_weight = ti.field(dtype=ti.i32, shape=TOTAL_POINTS)

# Simulation Parameters
gravity_val = ti.field(dtype=ti.f32, shape=())
puffiness = ti.field(dtype=ti.f32, shape=())
chord_len = ti.field(dtype=ti.f32, shape=())
target_area = ti.field(dtype=ti.f32, shape=())
circumference = ti.field(dtype=ti.f32, shape=())


# --- Taichi Helpers ---
@ti.func
def get_blob_area():
    area = 0.0
    for i in range(NUM_BLOB_POINTS):
        cur = pos[i]
        next_i = (i + 1) % NUM_BLOB_POINTS
        nxt = pos[next_i]
        area += (cur.x - nxt.x) * (cur.y + nxt.y) * 0.5
    # CW winding means area is naturally negative with this formula. By negating it, we get positive.
    return -area


# --- Taichi Kernels ---


@ti.kernel
def init_frog_kernel(origin: ti.math.vec2, radius: float, p_puff: float):
    puffiness[None] = p_puff

    # 1. 计算放松状态下（puffiness=1.0）的基础面积和真实边长
    base_area = (NUM_BLOB_POINTS / 2.0) * (radius * radius) * ti.sin(2.0 * math.pi / NUM_BLOB_POINTS)
    base_chord = 2.0 * radius * ti.sin(math.pi / NUM_BLOB_POINTS)

    # 2. Puffiness 仅作用于目标面积（模拟充气）
    target_area[None] = base_area * p_puff

    # 3. 边长保持基础长度（模拟橡皮的拉扯阻力），这样 PBD 求解器就会让它鼓起来
    chord_len[None] = base_chord
    circumference[None] = base_chord * NUM_BLOB_POINTS

    # 生成初始多边形（为了防止第一帧约束瞬间发力过猛，我们按稍微膨胀一点的形状生成）
    visual_scale = ti.sqrt(p_puff) if p_puff > 1.0 else 1.0
    for i in range(NUM_BLOB_POINTS):
        angle = (math.pi / 2.0) - (2.0 * math.pi * (i + 0.5) / NUM_BLOB_POINTS)
        offset = ti.Vector([ti.cos(angle), ti.sin(angle)]) * (radius * visual_scale)
        pos[i] = origin + offset
        ppos[i] = pos[i]


@ti.kernel
def substep_kernel(boundary_w: float, iterations: int):
    # 1. Verlet Integration & Gravity
    g_substep = gravity_val[None] * 0.001
    for i in range(TOTAL_POINTS):
        vel = (pos[i] - ppos[i]) * 0.99
        ppos[i] = pos[i]
        pos[i] += vel + ti.Vector([0.0, g_substep])

    # 2. Constraints (Blob)
    # The dummy loop ensures the internal iterations run sequentially, preventing race conditions on `disp`
    for _dummy in range(1):
        for _it in range(iterations):
            # Clear accumulation buffers
            for i in range(TOTAL_POINTS):
                disp[i] = ti.Vector([0.0, 0.0])
                disp_weight[i] = 0

            # Distance constraints
            for i in range(NUM_BLOB_POINTS):
                next_i = (i + 1) % NUM_BLOB_POINTS
                diff = pos[next_i] - pos[i]
                d = diff.norm()
                if d > chord_len[None]:
                    error = (d - chord_len[None]) * 0.5
                    if d > 1e-5:
                        offset = (diff / d) * error
                        # Accumulate
                        disp[i] += offset
                        disp_weight[i] += 1
                        disp[next_i] -= offset
                        disp_weight[next_i] += 1

            # Dilation constraints (Area)
            curr_area = get_blob_area()
            error_area = target_area[None] - curr_area
            # Limit the offset to avoid explosion on severe collapse/intersection
            offset_mag = ti.max(-0.5, ti.min(0.5, error_area / circumference[None]))

            for i in range(NUM_BLOB_POINTS):
                prev_i = (i - 1 + NUM_BLOB_POINTS) % NUM_BLOB_POINTS
                next_i = (i + 1) % NUM_BLOB_POINTS
                secant = pos[next_i] - pos[prev_i]
                d_sec = secant.norm()
                if d_sec > 1e-5:
                    # Rotated 90 deg CCW gives outward normal for CW winding
                    normal = ti.Vector([-secant.y, secant.x]) / d_sec
                    # Accumulate
                    disp[i] += normal * offset_mag
                    disp_weight[i] += 1

            # Apply accumulated displacements
            for i in range(TOTAL_POINTS):
                if disp_weight[i] > 0:
                    pos[i] += disp[i] / float(disp_weight[i])

            # Boundary (带有反弹和摩擦力的 Verlet 碰撞)
            restitution = 0.6  # 弹力系数 (0.0=粘滞, 1.0=完美弹性)
            friction = 0.95  # 摩擦力系数 (越小越容易打滑)

            for i in range(TOTAL_POINTS):
                # 左右边界
                if pos[i].x < -boundary_w / 2.0:
                    vel_x = pos[i].x - ppos[i].x
                    pos[i].x = -boundary_w / 2.0
                    ppos[i].x = pos[i].x + vel_x * restitution

                if pos[i].x > boundary_w / 2.0:
                    vel_x = pos[i].x - ppos[i].x
                    pos[i].x = boundary_w / 2.0
                    ppos[i].x = pos[i].x + vel_x * restitution

                # 地面碰撞 (核心弹跳逻辑)
                if pos[i].y < 0.0:
                    # 1. 记录撞击瞬间的隐式速度
                    vel_y = pos[i].y - ppos[i].y
                    vel_x = pos[i].x - ppos[i].x

                    # 2. 修正穿模的位置
                    pos[i].y = 0.0

                    # 3. 反转垂直速度（因为 vel_y 是负的，加上去会让 ppos 变负，从而在下一帧产生正向速度）
                    ppos[i].y = pos[i].y + vel_y * restitution

                    # 4. 在水平方向施加摩擦力减速
                    ppos[i].x = pos[i].x - vel_x * friction


# --- Blender Integration ---


def update_frog(scene):
    obj_name = scene.get("taichi_frog_obj", "")
    if not obj_name or obj_name not in bpy.data.objects:
        return

    obj = bpy.data.objects[obj_name]
    gravity_val[None] = scene.taichi_frog_gravity

    if scene.frame_current <= scene.frame_start:
        init_frog_kernel(ti.Vector([0, 5.0]), 1.28, scene.taichi_frog_puffiness)

    substep_kernel(20.0, scene.taichi_frog_substeps)

    data = pos.to_numpy()
    coords = np.zeros((TOTAL_POINTS, 3), dtype=np.float32)
    coords[:, 0], coords[:, 2] = data[:, 0], data[:, 1]

    obj.data.vertices.foreach_set("co", coords.flatten())
    obj.data.update()


@bpy.app.handlers.persistent
def frame_handler(scene):
    update_frog(scene)


class TAI_OT_init_frog(bpy.types.Operator):
    bl_idname = "taichi.init_frog"
    bl_label = "Initialize Simple Blob"

    def execute(self, context):
        mesh = bpy.data.meshes.new("TaichiFrog_Mesh")

        # Only the blob edges
        edges = [(i, (i + 1) % NUM_BLOB_POINTS) for i in range(NUM_BLOB_POINTS)]

        mesh.from_pydata(np.zeros((TOTAL_POINTS, 3)), edges, [])
        obj = bpy.data.objects.new("TaichiFrog", mesh)
        context.collection.objects.link(obj)
        context.scene["taichi_frog_obj"] = obj.name

        init_frog_kernel(ti.Vector([0, 5.0]), 1.28, context.scene.taichi_frog_puffiness)
        context.scene.frame_set(context.scene.frame_start)
        return {"FINISHED"}


class TAI_PT_frog_panel(bpy.types.Panel):
    bl_label = "Taichi Frog"
    bl_idname = "TAI_PT_frog_panel"
    bl_space_type, bl_region_type, bl_category = "VIEW_3D", "UI", "Taichi"

    def draw(self, context):
        layout, s = self.layout, context.scene
        layout.operator("taichi.init_frog", icon="OUTLINER_OB_MESH")
        layout.separator()
        layout.prop(s, "taichi_frog_puffiness")
        layout.prop(s, "taichi_frog_gravity")
        layout.prop(s, "taichi_frog_substeps")


def register():
    bpy.utils.register_class(TAI_OT_init_frog)
    bpy.utils.register_class(TAI_PT_frog_panel)
    bpy.types.Scene.taichi_frog_puffiness = bpy.props.FloatProperty(name="Puffiness", default=1.0, min=0.1, max=5.0)
    bpy.types.Scene.taichi_frog_gravity = bpy.props.FloatProperty(name="Gravity", default=-10.0, min=-100.0, max=100.0)
    bpy.types.Scene.taichi_frog_substeps = bpy.props.IntProperty(name="Substeps", default=10, min=1, max=50)
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre if h.__name__ != frame_handler.__name__
    ]
    if frame_handler not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(frame_handler)


def unregister():
    if frame_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(frame_handler)
    if hasattr(bpy.types.Scene, "taichi_frog_puffiness"):
        del bpy.types.Scene.taichi_frog_puffiness
    if hasattr(bpy.types.Scene, "taichi_frog_gravity"):
        del bpy.types.Scene.taichi_frog_gravity
    if hasattr(bpy.types.Scene, "taichi_frog_substeps"):
        del bpy.types.Scene.taichi_frog_substeps
    bpy.utils.unregister_class(TAI_PT_frog_panel)
    bpy.utils.unregister_class(TAI_OT_init_frog)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
