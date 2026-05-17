import bpy
import numpy as np
import taichi as ti

# Safe initialization: guard ti.init() for hot-reload compatibility
if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)


# ==========================================
# 1. 物理状态
# ==========================================
dg_state = {}


# ==========================================
# 2. Taichi 计算核心 (培养皿边界 + 严格 2D)
# ==========================================
@ti.kernel
def pbd_step_kernel(
    pos: ti.types.ndarray(dtype=ti.f32, ndim=2),
    pos_next: ti.types.ndarray(dtype=ti.f32, ndim=2),
    n: ti.i32,
    rep_radius: ti.f32,
    rep_weight: ti.f32,
    smooth_weight: ti.f32,
    bound_radius: ti.f32,
    bound_weight: ti.f32,
    flatten: ti.i32,
):
    for i in range(n):
        disp = ti.Vector([0.0, 0.0, 0.0])
        p_i = ti.Vector([pos[i, 0], pos[i, 1], pos[i, 2]])

        # 1. 弹簧/平滑约束 (维持线条的连续性)
        prev_i = (i - 1 + n) % n
        next_i = (i + 1) % n
        p_prev = ti.Vector([pos[prev_i, 0], pos[prev_i, 1], pos[prev_i, 2]])
        p_next = ti.Vector([pos[next_i, 0], pos[next_i, 1], pos[next_i, 2]])

        mid = (p_prev + p_next) * 0.5
        disp += (mid - p_i) * smooth_weight

        # 2. 局部与全局排斥 (促发自发扭曲的核心动力)
        rep_disp = ti.Vector([0.0, 0.0, 0.0])
        for j in range(n):
            if i != j:
                p_j = ti.Vector([pos[j, 0], pos[j, 1], pos[j, 2]])
                diff = p_i - p_j
                dist = diff.norm()
                if dist < rep_radius and dist > 1e-5:
                    rep_disp += diff.normalized() * (rep_radius - dist)
        disp += rep_disp * rep_weight

        # 3. 培养皿边界约束 (把刺头按回来)
        if bound_radius > 0.0:
            # 如果开启了严格 2D，边界计算也只看 XY 距离
            dist_center = p_i.norm()
            if flatten == 1:
                dist_center = ti.math.sqrt(p_i[0] * p_i[0] + p_i[1] * p_i[1])

            if dist_center > bound_radius:
                bound_dir = -p_i.normalized()
                if flatten == 1:
                    bound_dir[2] = 0.0  # 确保边界推力不会产生 Z 轴位移
                disp += bound_dir * (dist_center - bound_radius) * bound_weight

        # 限制单步最大位移以保证物理稳定
        disp_norm = disp.norm()
        if disp_norm > rep_radius * 0.5:
            disp = disp.normalized() * (rep_radius * 0.1)

        # 4. 严格 2D 拍扁逻辑 (flatten == 1 时锁定 Z=0)
        if flatten == 1:
            pos_next[i, 0] = p_i[0] + disp[0]
            pos_next[i, 1] = p_i[1] + disp[1]
            pos_next[i, 2] = 0.0  # 绝对锁定 Z=0
        else:
            pos_next[i, 0] = p_i[0] + disp[0]
            pos_next[i, 1] = p_i[1] + disp[1]
            pos_next[i, 2] = p_i[2] + disp[2]


# ==========================================
# 3. Python 拓扑生长逻辑
# ==========================================
def grow_topology(pos, split_dist, max_splits, flatten):
    n = len(pos)
    edges = []

    for i in range(n):
        next_i = (i + 1) % n
        dist = np.linalg.norm(pos[next_i] - pos[i])
        if dist > split_dist:
            edges.append((dist, i, next_i))

    if not edges:
        return pos

    # 按长度降序排列，优先分裂最长的边
    edges.sort(key=lambda x: x[0], reverse=True)
    splits = edges[:max_splits]
    split_indices = set(e[1] for e in splits)

    new_pos = []
    for i in range(n):
        new_pos.append(pos[i])
        if i in split_indices:
            next_i = (i + 1) % n
            mid_p = (pos[i] + pos[next_i]) * 0.5

            noise = (np.random.rand(3) - 0.5) * (split_dist * 0.1)
            if flatten:
                noise[2] = 0.0

            mid_p += noise
            new_pos.append(mid_p)

    return np.array(new_pos, dtype=np.float32)


def update_blender_mesh(obj, pos):
    """Update target mesh. Structural ops (clear_geometry/from_pydata) are gated to Object Mode."""
    mesh = obj.data
    n = len(pos)

    if len(mesh.vertices) != n:
        # Structural operation — must be in Object Mode
        if obj.mode != 'OBJECT':
            return
        edges = [(i, (i + 1) % n) for i in range(n)]
        mesh.clear_geometry()
        mesh.from_pydata(pos.tolist(), edges, [])
    else:
        # Fast path — safe in Object Mode
        mesh.vertices.foreach_set("co", pos.flatten())

    mesh.update()


# ==========================================
# 4. 动画帧处理器
# ==========================================
def reset_dg_simulation(scene, obj):
    global dg_state
    props = scene.taichi_dg_props
    n = 20
    radius = 1.0
    pos = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        angle = 2.0 * np.pi * i / n
        pos[i, 0] = np.cos(angle) * radius
        pos[i, 1] = np.sin(angle) * radius
        if not props.flatten:
            pos[i, 2] += (np.random.rand() - 0.5) * 0.1

        pos[i, 0] += (np.random.rand() - 0.5) * 0.1
        pos[i, 1] += (np.random.rand() - 0.5) * 0.1

    dg_state = {"pos": pos}
    update_blender_mesh(obj, pos)


@bpy.app.handlers.persistent
def dg_frame_handler(scene, depsgraph):
    obj_name = scene.get("taichi_dg_obj", "")
    if not obj_name or obj_name not in bpy.data.objects:
        return

    obj = bpy.data.objects[obj_name]

    # Mode safety: skip update if target is in Edit Mode
    if obj.mode != 'OBJECT':
        return

    props = scene.taichi_dg_props

    if scene.frame_current <= scene.frame_start:
        reset_dg_simulation(scene, obj)
        return

    global dg_state
    if not dg_state:
        reset_dg_simulation(scene, obj)

    pos = dg_state["pos"]
    n = len(pos)
    max_nodes = props.max_nodes
    substeps = props.substeps
    pos_next = np.zeros_like(pos)

    flatten = 1 if props.flatten else 0

    for _ in range(substeps):
        pbd_step_kernel(
            pos,
            pos_next,
            n,
            props.rep_radius,
            props.rep_weight,
            props.smooth_weight,
            props.bound_radius,
            props.bound_weight,
            flatten,
        )
        pos[:] = pos_next[:]

    if n < max_nodes:
        pos = grow_topology(pos, props.split_dist, props.growth_rate, flatten)

    dg_state["pos"] = pos
    update_blender_mesh(obj, pos)


# ==========================================
# 5. PropertyGroup
# ==========================================
class TaichiDG_Properties(bpy.types.PropertyGroup):
    max_nodes: bpy.props.IntProperty(name="Max Nodes", default=5000, min=100, max=20000)
    substeps: bpy.props.IntProperty(name="Substeps", default=10, min=1, max=50)
    growth_rate: bpy.props.IntProperty(name="Growth Rate (Splits/Frame)", default=3, min=1, max=50)
    split_dist: bpy.props.FloatProperty(name="Split Distance", default=0.15, min=0.01)
    rep_radius: bpy.props.FloatProperty(name="Repel Radius", default=0.4, min=0.05)
    rep_weight: bpy.props.FloatProperty(name="Repel Weight", default=0.2, min=0.0)
    smooth_weight: bpy.props.FloatProperty(name="Smooth Weight", default=0.4, min=0.0)
    bound_radius: bpy.props.FloatProperty(
        name="Boundary Radius", default=4.0, min=0.0,
        description="Limits the maximum outward growth, forcing inward folding.",
    )
    bound_weight: bpy.props.FloatProperty(name="Boundary Rigidity", default=0.5, min=0.0)
    flatten: bpy.props.BoolProperty(name="Strictly Constrain to Z=0", default=True)


# ==========================================
# 6. UI 操作符与面板
# ==========================================
class TAI_OT_create_dg(bpy.types.Operator):
    bl_idname = "taichi.create_dg"
    bl_label = "Init Pure Differential Growth"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mesh = bpy.data.meshes.new("PureGrowth_Curve")
        obj = bpy.data.objects.new("Taichi_Pure_Growth", mesh)
        context.collection.objects.link(obj)

        context.scene["taichi_dg_obj"] = obj.name
        context.scene.frame_set(context.scene.frame_start)

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj

        return {"FINISHED"}


class TAI_PT_dg_panel(bpy.types.Panel):
    bl_label = "Pure Growth (Taichi)"
    bl_idname = "TAI_PT_dg_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.taichi_dg_props

        layout.operator(TAI_OT_create_dg.bl_idname, icon="OUTLINER_OB_CURVE")

        if scene.get("taichi_dg_obj", "") in bpy.data.objects:
            layout.separator()
            box = layout.box()
            box.label(text="Simulation Control:")
            box.prop(props, "max_nodes")
            box.prop(props, "substeps")

            box2 = layout.box()
            box2.label(text="Growth Parameters:")
            box2.prop(props, "growth_rate")
            box2.prop(props, "split_dist")

            box3 = layout.box()
            box3.label(text="PBD Constraints:")
            box3.prop(props, "rep_radius")
            box3.prop(props, "rep_weight")
            box3.prop(props, "smooth_weight")
            box3.prop(props, "bound_radius")
            box3.prop(props, "bound_weight")

            layout.separator()
            layout.prop(props, "flatten", icon="EXPORT")

            # Data Interface Disclosure
            layout.separator()
            box_di = layout.box()
            box_di.label(text="Data Interface:")
            row = box_di.row()
            row.label(text="Attribute")
            row.label(text="Domain")
            row = box_di.row()
            row.label(text="position")
            row.label(text="Point")
            row = box_di.row()
            row.label(text="edge_topology")
            row.label(text="Edge")


# ==========================================
# 7. 注册
# ==========================================
classes = (
    TaichiDG_Properties,
    TAI_OT_create_dg,
    TAI_PT_dg_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.taichi_dg_props = bpy.props.PointerProperty(type=TaichiDG_Properties)

    # Surgical handler cleanup by name to prevent stacking
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre
        if h.__name__ != dg_frame_handler.__name__
    ]
    bpy.app.handlers.frame_change_pre.append(dg_frame_handler)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.taichi_dg_props

    if dg_frame_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(dg_frame_handler)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
