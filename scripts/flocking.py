# https://zalo.github.io/blog/constraints/

import time

import bpy
import numpy as np
import taichi as ti
from bpy_extras import view3d_utils

# =========================
# 1. Taichi 计算核心 (Boids Algorithm)
# =========================

if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu, log_level=ti.INFO)
    except:
        ti.init(arch=ti.cpu)

MAX_BOIDS = 10000
pos = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BOIDS)
vel = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BOIDS)
acc = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BOIDS)


@ti.kernel
def init_flock(n: ti.i32, width: ti.f32, height: ti.f32):
    for i in range(n):
        pos[i] = [ti.random() * width - width / 2, ti.random() * height - height / 2]
        angle = ti.random() * 2.0 * np.pi
        vel[i] = [ti.cos(angle), ti.sin(angle)]
        acc[i] = [0.0, 0.0]


@ti.kernel
def spawn_boid_kernel(idx: ti.i32, x: ti.f32, y: ti.f32):
    pos[idx] = [x, y]
    angle = ti.random() * 2.0 * np.pi
    vel[idx] = [ti.cos(angle), ti.sin(angle)]
    acc[idx] = [0.0, 0.0]


@ti.func
def steer_towards(i, target_vel, max_speed, max_force):
    steer = ti.Vector([0.0, 0.0])
    if target_vel.norm() > 0:
        desired = target_vel.normalized() * max_speed
        steer = desired - vel[i]
        if steer.norm() > max_force:
            steer = steer.normalized() * max_force
    return steer


@ti.kernel
def compute_boids(
    n: ti.i32,
    max_speed: ti.f32,
    max_force: ti.f32,
    sep_rad: ti.f32,
    ali_rad: ti.f32,
    coh_rad: ti.f32,
    sep_weight: ti.f32,
    ali_weight: ti.f32,
    coh_weight: ti.f32,
    width: ti.f32,
    height: ti.f32,
    dt: ti.f32,
):
    for i in range(n):
        sep_steer = ti.Vector([0.0, 0.0])
        ali_sum = ti.Vector([0.0, 0.0])
        coh_sum = ti.Vector([0.0, 0.0])
        sep_count, ali_count, coh_count = 0, 0, 0

        for j in range(n):
            if i == j:
                continue
            diff = pos[i] - pos[j]
            d = diff.norm()
            if d > 0 and d < sep_rad:
                sep_steer += diff.normalized() / (d + 1e-4)
                sep_count += 1
            if d > 0 and d < ali_rad:
                ali_sum += vel[j]
                ali_count += 1
            if d > 0 and d < coh_rad:
                coh_sum += pos[j]
                coh_count += 1

        if sep_count > 0:
            sep_steer /= float(sep_count)
            if sep_steer.norm() > 0:
                sep_steer = sep_steer.normalized() * max_speed - vel[i]
                if sep_steer.norm() > max_force:
                    sep_steer = sep_steer.normalized() * max_force

        ali_steer = steer_towards(
            i, ali_sum / float(ali_count) if ali_count > 0 else ti.Vector([0.0, 0.0]), max_speed, max_force
        )
        coh_steer = steer_towards(
            i, (coh_sum / float(coh_count)) - pos[i] if coh_count > 0 else ti.Vector([0.0, 0.0]), max_speed, max_force
        )

        acc[i] = sep_steer * sep_weight + ali_steer * ali_weight + coh_steer * coh_weight

    w_half, h_half = width / 2.0, height / 2.0
    for i in range(n):
        vel[i] += acc[i] * dt
        if vel[i].norm() > max_speed:
            vel[i] = vel[i].normalized() * max_speed
        pos[i] += vel[i] * dt
        if pos[i].x < -w_half:
            pos[i].x = w_half
        if pos[i].x > w_half:
            pos[i].x = -w_half
        if pos[i].y < -h_half:
            pos[i].y = h_half
        if pos[i].y > h_half:
            pos[i].y = -h_half


# =========================
# 2. Properties & State
# =========================


class FlockingSettings(bpy.types.PropertyGroup):
    count: bpy.props.IntProperty(name="Boid Count", default=500, min=0, max=MAX_BOIDS)
    substeps: bpy.props.IntProperty(name="Substeps", default=2, min=1, max=10)
    max_speed: bpy.props.FloatProperty(name="Max Speed", default=0.2, min=0.01)
    max_force: bpy.props.FloatProperty(name="Max Force", default=0.01, min=0.001)

    sep_rad: bpy.props.FloatProperty(name="Separation Radius", default=0.5, min=0.1)
    ali_rad: bpy.props.FloatProperty(name="Alignment Radius", default=1.0, min=0.1)
    coh_rad: bpy.props.FloatProperty(name="Cohesion Radius", default=1.0, min=0.1)

    sep_weight: bpy.props.FloatProperty(name="Separation Weight", default=1.5, min=0.0)
    ali_weight: bpy.props.FloatProperty(name="Alignment Weight", default=1.0, min=0.0)
    coh_weight: bpy.props.FloatProperty(name="Cohesion Weight", default=1.0, min=0.0)

    width: bpy.props.FloatProperty(name="Width", default=20.0, min=1.0)
    height: bpy.props.FloatProperty(name="Height", default=20.0, min=1.0)

    # State tracking
    is_tracking: bpy.props.BoolProperty(name="Is Tracking", default=False)
    is_dragging: bpy.props.BoolProperty(name="Is Dragging", default=False)
    target_obj: bpy.props.PointerProperty(name="Target Object", type=bpy.types.Object)


# =========================
# 3. Geometry Nodes Helper
# =========================


def setup_flocking_gn(obj):
    group_name = "Flocking_Visualizer"
    node_group = bpy.data.node_groups.get(group_name)

    if node_group:
        bpy.data.node_groups.remove(node_group)

    node_group = bpy.data.node_groups.new(type="GeometryNodeTree", name=group_name)
    node_group.is_modifier = True

    node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")

    nodes = node_group.nodes
    links = node_group.links

    inp = nodes.new("NodeGroupInput")
    out = nodes.new("NodeGroupOutput")
    inp.location = (-600, 200)
    out.location = (600, 200)

    inst = nodes.new("GeometryNodeInstanceOnPoints")
    inst.location = (200, 200)
    inst.inputs[6].default_value = (2.0, 1.0, 1.0)

    circle = nodes.new("GeometryNodeMeshCircle")
    circle.fill_type = "TRIANGLE_FAN"
    circle.inputs[0].default_value = 3
    circle.inputs[1].default_value = 0.1
    circle.location = (-100, 100)

    attr_rot = nodes.new("GeometryNodeInputNamedAttribute")
    attr_rot.data_type = "FLOAT_VECTOR"
    attr_rot.inputs[0].default_value = "velocity"
    attr_rot.location = (-400, 0)

    align = nodes.new("FunctionNodeAlignRotationToVector")
    align.axis = "X"
    align.location = (-100, -50)

    viewer = nodes.new("GeometryNodeViewer")
    viewer.domain = "INSTANCE"
    viewer.location = (600, -100)
    viewer.viewer_items.clear()
    viewer.viewer_items.new("GEOMETRY", "Instances")
    viewer.viewer_items.new("VECTOR", "Vector")

    attr_color = nodes.new("GeometryNodeInputNamedAttribute")
    attr_color.data_type = "FLOAT_VECTOR"
    attr_color.inputs[0].default_value = "velocity"
    attr_color.location = (-400, -250)

    vec_abs = nodes.new("ShaderNodeVectorMath")
    vec_abs.operation = "ABSOLUTE"
    vec_abs.location = (-100, -250)

    vec_norm = nodes.new("ShaderNodeVectorMath")
    vec_norm.operation = "NORMALIZE"
    vec_norm.location = (150, -250)

    links.new(inp.outputs[0], inst.inputs[0])
    links.new(circle.outputs[0], inst.inputs[2])
    links.new(attr_rot.outputs[0], align.inputs[2])
    links.new(align.outputs[0], inst.inputs[5])
    links.new(inst.outputs[0], out.inputs[0])

    links.new(inst.outputs[0], viewer.inputs[0])
    links.new(attr_color.outputs[0], vec_abs.inputs[0])
    links.new(vec_abs.outputs[0], vec_norm.inputs[0])
    links.new(vec_norm.outputs[0], viewer.inputs[1])

    mod = obj.modifiers.get("Taichi_Flocking_GN")
    if not mod:
        mod = obj.modifiers.new("Taichi_Flocking_GN", "NODES")
    mod.node_group = node_group


# =========================
# 4. Blender 交互逻辑
# =========================


def sync_mesh_data(scene):
    props = scene.taichi_flocking
    obj = props.target_obj
    if not obj:
        return
    mesh = obj.data

    n = props.count
    if n == 0:
        mesh.clear_geometry()
        return

    pos_np = pos.to_numpy()[:n]
    vel_np = vel.to_numpy()[:n]

    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:, 0], coords[:, 1] = pos_np[:, 0], pos_np[:, 1]

    vel_attr = np.zeros((n, 3), dtype=np.float32)
    vel_attr[:, 0], vel_attr[:, 1] = vel_np[:, 0], vel_np[:, 1]

    if len(mesh.vertices) != n:
        mesh.clear_geometry()
        mesh.vertices.add(n)

    mesh.vertices.foreach_set("co", coords.flatten())
    if "velocity" not in mesh.attributes:
        mesh.attributes.new(name="velocity", type="FLOAT_VECTOR", domain="POINT")
    mesh.attributes["velocity"].data.foreach_set("vector", vel_attr.flatten())
    mesh.update()
    ti.sync()


def run_flocking_frame(scene):
    props = scene.taichi_flocking
    n = props.count
    if scene.frame_current <= scene.frame_start:
        init_flock(n, props.width, props.height)
    else:
        for _ in range(props.substeps):
            compute_boids(
                n,
                props.max_speed,
                props.max_force,
                props.sep_rad,
                props.ali_rad,
                props.coh_rad,
                props.sep_weight,
                props.ali_weight,
                props.coh_weight,
                props.width,
                props.height,
                0.1,
            )
    sync_mesh_data(scene)


@bpy.app.handlers.persistent
def flocking_update_handler(scene):
    props = scene.taichi_flocking
    is_playing = bpy.context.screen and bpy.context.screen.is_animation_playing
    if props.is_tracking or is_playing:
        run_flocking_frame(scene)


class TAI_OT_flocking_modal(bpy.types.Operator):
    bl_idname = "taichi.flocking_modal"
    bl_label = "Interactive Flocking"

    _last_spawn_time = 0.0

    def modal(self, context, event):
        # 0. 基础安全检查：确保环境有效
        if not context.area or not hasattr(context.scene, "taichi_flocking"):
            return {"PASS_THROUGH"}

        props = context.scene.taichi_flocking
        context.area.tag_redraw()

        # Exit if toggled off from UI or ESC/RightMouse pressed
        if not props.is_tracking or event.type in {"ESC", "RIGHTMOUSE"}:
            props.is_tracking = False
            props.is_dragging = False
            if props.target_obj:
                props.target_obj.hide_select = False
            return {"FINISHED"}

        # 1. 投影计算：确定鼠标是否在仿真平面及区域内
        # 寻找当前区域内的 WINDOW region (3D 视口主区)
        region = next((r for r in context.area.regions if r.type == "WINDOW"), None)
        rv3d = context.space_data.region_3d if hasattr(context.space_data, "region_3d") else None

        is_over_sim = False
        hit_pos = None
        if region and rv3d:
            # 只有当鼠标在视口主区域内时才进行坐标转换
            if (
                region.x <= event.mouse_x <= region.x + region.width
                and region.y <= event.mouse_y <= region.y + region.height
            ):
                coord = (event.mouse_x - region.x, event.mouse_y - region.y)
                origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
                direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

                if abs(direction.z) > 1e-6:
                    hit = origin + (-origin.z / direction.z) * direction
                    if abs(hit.x) <= props.width / 2.0 and abs(hit.y) <= props.height / 2.0:
                        is_over_sim = True
                        hit_pos = hit

        # 2. 鼠标左键逻辑
        if event.type == "LEFTMOUSE":
            if event.value == "PRESS":
                if is_over_sim:
                    props.is_dragging = True
                    return {"RUNNING_MODAL"}  # 仅在区域内拦截点击
            elif event.value == "RELEASE":
                if props.is_dragging:
                    props.is_dragging = False
                    return {"RUNNING_MODAL"}  # 拦截正在拖拽的释放
            return {"PASS_THROUGH"}  # 其他情况允许正常选择

        # 3. 鼠标移动逻辑
        elif event.type == "MOUSEMOVE":
            if props.is_dragging:
                # 只有在区域内才生成顶点
                if is_over_sim and hit_pos:
                    current_time = time.time()
                    if current_time - self.__class__._last_spawn_time > 0.05:
                        if props.count < MAX_BOIDS:
                            spawn_boid_kernel(props.count, hit_pos.x, hit_pos.y)
                            props.count += 1
                            self.__class__._last_spawn_time = current_time
                            if not (bpy.context.screen and bpy.context.screen.is_animation_playing):
                                sync_mesh_data(context.scene)
                return {"RUNNING_MODAL"}  # 正在拖拽时拦截移动防止框选

        return {"PASS_THROUGH"}

    def invoke(self, context, event):
        props = context.scene.taichi_flocking
        if context.space_data.type == "VIEW_3D":
            props.is_tracking = True
            props.is_dragging = False
            if props.target_obj:
                props.target_obj.hide_select = True
                props.target_obj.select_set(False)
            context.window_manager.modal_handler_add(self)
            return {"RUNNING_MODAL"}
        return {"CANCELLED"}


class TAI_OT_create_flock(bpy.types.Operator):
    bl_idname = "taichi.create_flock"
    bl_label = "Create Flocking System"

    def execute(self, context):
        props = context.scene.taichi_flocking
        obj = bpy.data.objects.get("Taichi_Flock")
        if not obj:
            mesh = bpy.data.meshes.new("Flock_Mesh")
            obj = bpy.data.objects.new("Taichi_Flock", mesh)
            context.collection.objects.link(obj)
        props.target_obj = obj
        setup_flocking_gn(obj)
        init_flock(props.count, props.width, props.height)
        run_flocking_frame(context.scene)
        return {"FINISHED"}


class TAI_PT_flocking_panel(bpy.types.Panel):
    bl_label = "Taichi Flocking"
    bl_idname = "TAI_PT_flocking_panel"
    bl_space_type, bl_region_type, bl_category = "VIEW_3D", "UI", "Taichi"

    def draw(self, context):
        layout, props = self.layout, context.scene.taichi_flocking
        if not props.is_tracking:
            layout.operator("taichi.flocking_modal", text="Start Interaction", icon="PLAY")
        else:
            layout.prop(props, "is_tracking", text="Stop Interaction (ESC)", icon="CANCEL", toggle=True)
            layout.label(
                text="Status: 🟢 DRAGGING" if props.is_dragging else "Status: 🔵 WAITING CLICK",
                icon="MOUSE_MOVE" if props.is_dragging else "MOUSE_LMB",
            )
        layout.operator("taichi.create_flock", icon="PARTICLE_DATA")
        if props.target_obj:
            box = layout.box()
            box.label(text="Simulation Settings")
            box.prop(props, "count")
            box.prop(props, "substeps")
            box = layout.box()
            box.label(text="Physical Constants")
            for p in ["max_speed", "max_force", "width", "height"]:
                box.prop(props, p)
            box = layout.box()
            box.label(text="Behavior Weights")
            for p in ["sep_weight", "ali_weight", "coh_weight"]:
                box.prop(props, p)
            box = layout.box()
            box.label(text="Perception Radii")
            for p in ["sep_rad", "ali_rad", "coh_rad"]:
                box.prop(props, p)


def register():
    bpy.utils.register_class(FlockingSettings)
    bpy.utils.register_class(TAI_OT_create_flock)
    bpy.utils.register_class(TAI_OT_flocking_modal)
    bpy.utils.register_class(TAI_PT_flocking_panel)
    bpy.types.Scene.taichi_flocking = bpy.props.PointerProperty(type=FlockingSettings)
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre if h.__name__ != flocking_update_handler.__name__
    ]
    bpy.app.handlers.frame_change_pre.append(flocking_update_handler)


def unregister():
    if flocking_update_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(flocking_update_handler)
    del bpy.types.Scene.taichi_flocking
    for cls in reversed([FlockingSettings, TAI_OT_create_flock, TAI_OT_flocking_modal, TAI_PT_flocking_panel]):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
