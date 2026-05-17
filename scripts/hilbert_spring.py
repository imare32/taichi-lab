import bpy
import numpy as np
import taichi as ti

# Safe initialization for Blender hot-reloading
if not getattr(ti.lang.impl.get_runtime(), "prog", None):
    try:
        ti.init(arch=ti.gpu)
    except:
        ti.init(arch=ti.cpu)


# =========================
# 核心参数与状态 (Taichi 字段分配)
# =========================
MAX_ORDER = 4  # 最大支持 4 阶 (4096 点)
MAX_VERTICES = (1 << MAX_ORDER)**3
MAX_SPRINGS = MAX_VERTICES * 26 // 2  # 每个点邻居上限

# 声明 Taichi 场 (单例模式)
pos = ti.Vector.field(3, dtype=ti.f64, shape=MAX_VERTICES)
vel = ti.Vector.field(3, dtype=ti.f64, shape=MAX_VERTICES)
force = ti.Vector.field(3, dtype=ti.f64, shape=MAX_VERTICES)
spring_pairs = ti.Vector.field(2, dtype=ti.i32, shape=MAX_SPRINGS)
spring_lengths = ti.field(dtype=ti.f64, shape=MAX_SPRINGS)

num_v_active = ti.field(dtype=ti.i32, shape=())
num_s_active = ti.field(dtype=ti.i32, shape=())

# 物理常数 (由 Operator 更新)
stiffness = ti.field(dtype=ti.f64, shape=())
damping = ti.field(dtype=ti.f64, shape=())
dt_val = ti.field(dtype=ti.f64, shape=())
mass_val = ti.field(dtype=ti.f64, shape=())

# 缓存数据用于重置
cached_data = {
    "num_vertices": 0,
    "init_pos": None,
    "spring_pairs": None,
    "spring_lengths": None
}

# =========================
# 1. 希尔伯特映射与网格算法
# =========================
def hilbert_3d_point(index, order):
    n = 3
    bits = format(index, f"0{order * n}b")
    x = [int(bits[i::n], 2) if bits[i::n] else 0 for i in range(n)]
    t = x[n - 1] >> 1
    for i in range(n - 1, 0, -1):
        x[i] ^= x[i - 1]
    x[0] ^= t
    q = 2
    z = 1 << order
    while q != z:
        p = q - 1
        for i in range(n - 1, -1, -1):
            if x[i] & q:
                x[0] ^= p
            else:
                t = (x[0] ^ x[i]) & p
                x[0] ^= t
                x[i] ^= t
        q <<= 1
    return tuple(x)


# =========================
# 2. Taichi 物理引擎封装
# =========================

def prepare_sim_data(order, size, drop_height):
    g_size = 1 << order
    n_verts = g_size**3
    step_size = size / (g_size - 1) if g_size > 1 else 1.0
    grid_to_index = {}
    half = size / 2.0

    init_p = np.zeros((n_verts, 3), dtype=np.float32)
    for i in range(n_verts):
        gx, gy, gz = hilbert_3d_point(i, order)
        grid_to_index[(gx, gy, gz)] = i
        init_p[i] = [gx * step_size - half, gy * step_size - half, gz * step_size - half + drop_height]

    sp_pairs = []
    sp_lengths = []
    offsets = [
        (dx, dy, dz)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    for gx in range(g_size):
        for gy in range(g_size):
            for gz in range(g_size):
                idx_a = grid_to_index[(gx, gy, gz)]
                for dx, dy, dz in offsets:
                    nx, ny, nz = gx + dx, gy + dy, gz + dz
                    if 0 <= nx < g_size and 0 <= ny < g_size and 0 <= nz < g_size:
                        idx_b = grid_to_index[(nx, ny, nz)]
                        if idx_a < idx_b:
                            sp_pairs.append([idx_a, idx_b])
                            p_a = init_p[idx_a]
                            p_b = init_p[idx_b]
                            length = np.linalg.norm(p_a - p_b)
                            sp_lengths.append(length)
    
    return n_verts, init_p, np.array(sp_pairs, dtype=np.int32), np.array(sp_lengths, dtype=np.float32)


@ti.kernel
def reset_sim_kernel(n_v: int, init_p: ti.types.ndarray(), n_s: int, sp_p: ti.types.ndarray(), sp_l: ti.types.ndarray()):
    num_v_active[None] = n_v
    num_s_active[None] = n_s
    for i in range(n_v):
        pos[i] = ti.Vector([init_p[i, 0], init_p[i, 1], init_p[i, 2]])
        vel[i] = ti.Vector([0.0, 0.0, 0.0])
    for i in range(n_s):
        spring_pairs[i] = ti.Vector([sp_p[i, 0], sp_p[i, 1]])
        spring_lengths[i] = sp_l[i]


@ti.kernel
def compute_forces():
    m = mass_val[None]
    stiff = stiffness[None]
    damp = damping[None]
    
    for i in range(num_v_active[None]):
        force[i] = ti.Vector([0.0, 0.0, -9.8]) * m

    for i in range(num_s_active[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        L0 = spring_lengths[i]

        p_a = pos[idx_a]
        p_b = pos[idx_b]
        v_a = vel[idx_a]
        v_b = vel[idx_b]

        diff = p_a - p_b
        current_L = diff.norm()

        if current_L > 1e-5:
            dir = diff / current_L
            f_spring = -stiff * (current_L - L0) * dir
            v_rel = v_a - v_b
            f_damp = -damp * v_rel.dot(dir) * dir
            f_total = f_spring + f_damp
            force[idx_a] += f_total
            force[idx_b] -= f_total


@ti.kernel
def advance():
    dt = dt_val[None]
    m = mass_val[None]
    for i in range(num_v_active[None]):
        vel[i] += (force[i] / m) * dt
        pos[i] += vel[i] * dt
        if pos[i][2] < 0.0:
            pos[i][2] = 0.0
            vel[i][2] *= -0.8
            vel[i][0] *= 0.7
            vel[i][1] *= 0.7


@ti.kernel
def export_to_blender(out_pos: ti.types.ndarray(), out_vmag: ti.types.ndarray()):
    for i in range(num_v_active[None]):
        p = pos[i]
        out_pos[i, 0] = ti.cast(p[0], ti.f32)
        out_pos[i, 1] = ti.cast(p[1], ti.f32)
        out_pos[i, 2] = ti.cast(p[2], ti.f32)

        v = vel[i]
        out_vmag[i] = ti.cast(v.norm(), ti.f32)


def step(substeps):
    for _ in range(substeps):
        compute_forces()
        advance()


def reset_simulation():
    if cached_data["init_pos"] is None:
        return
    
    reset_sim_kernel(
        cached_data["num_vertices"],
        cached_data["init_pos"],
        len(cached_data["spring_pairs"]),
        cached_data["spring_pairs"],
        cached_data["spring_lengths"]
    )


# =========================
# 3. Blender 交互与设置
# =========================

class TAI_HilbertProperties(bpy.types.PropertyGroup):
    order: bpy.props.IntProperty(
        name="Order", 
        description="Hilbert curve order (1-4)",
        default=3, min=1, max=MAX_ORDER
    )
    size: bpy.props.FloatProperty(
        name="Size", 
        description="Initial size of the curve",
        default=4.0, min=0.1
    )
    drop_height: bpy.props.FloatProperty(
        name="Drop Height", 
        description="Initial Z height",
        default=5.0, min=0.0
    )
    stiffness: bpy.props.FloatProperty(
        name="Stiffness", 
        description="Spring stiffness",
        default=10000.0, min=0.0
    )
    damping: bpy.props.FloatProperty(
        name="Damping", 
        description="Spring damping",
        default=15.0, min=0.0
    )
    mass: bpy.props.FloatProperty(
        name="Mass", 
        description="Vertex mass",
        default=1.0, min=0.01
    )
    dt: bpy.props.FloatProperty(
        name="Time Step", 
        description="Simulation time step",
        default=2e-4, precision=5, step=0.01
    )
    substeps: bpy.props.IntProperty(
        name="Substeps", 
        description="Physics substeps per frame",
        default=150, min=1
    )


def run_hilbert_frame(scene):
    obj_name = scene.get("taichi_hilbert_obj", "")
    if not obj_name or obj_name not in bpy.data.objects:
        return

    obj = bpy.data.objects[obj_name]
    if obj.mode != "OBJECT":
        return

    mesh = obj.data
    n_verts = cached_data["num_vertices"]
    if n_verts == 0:
        return

    if scene.frame_current <= scene.frame_start:
        reset_simulation()
    else:
        # 使用存储在场景属性中的步数
        substeps = scene.taichi_hilbert.substeps
        step(substeps)

    pos_np = np.zeros((n_verts, 3), dtype=np.float32)
    vmag_np = np.zeros(n_verts, dtype=np.float32)

    export_to_blender(pos_np, vmag_np)

    mesh.vertices.foreach_set("co", pos_np.flatten())
    if "velocity_mag" in mesh.attributes:
        mesh.attributes["velocity_mag"].data.foreach_set("value", vmag_np)

    mesh.update()
    ti.sync()


@bpy.app.handlers.persistent
def hilbert_update_handler(scene):
    run_hilbert_frame(scene)


class TAI_OT_create_hilbert(bpy.types.Operator):
    bl_idname = "taichi.create_hilbert"
    bl_label = "Spawn Hilbert Curve Simulation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.taichi_hilbert
        obj_name = "Hilbert_Simulation"
        
        # 更新物理参数
        stiffness[None] = props.stiffness
        damping[None] = props.damping
        mass_val[None] = props.mass
        dt_val[None] = props.dt

        # 准备数据
        n_v, init_p, sp_p, sp_l = prepare_sim_data(props.order, props.size, props.drop_height)
        cached_data["num_vertices"] = n_v
        cached_data["init_pos"] = init_p
        cached_data["spring_pairs"] = sp_p
        cached_data["spring_lengths"] = sp_l

        reset_simulation()

        # 复用或创建对象
        obj = bpy.data.objects.get(obj_name)
        if obj is None or obj.type != 'MESH':
            mesh = bpy.data.meshes.new(obj_name + "_Mesh")
            obj = bpy.data.objects.new(obj_name, mesh)
            context.collection.objects.link(obj)
        else:
            mesh = obj.data
            # 如果不是当前集合的，重新链接（可选，但安全）
            if obj.name not in context.collection.objects:
                context.collection.objects.link(obj)
        
        # 更新几何结构 (拓扑可能随 Order 改变)
        edges = [(i, i + 1) for i in range(n_v - 1)]
        mesh.clear_geometry()
        mesh.from_pydata(init_p, edges, [])

        # 确保属性存在
        if "velocity_mag" not in mesh.attributes:
            mesh.attributes.new(name="velocity_mag", type="FLOAT", domain="POINT")
        
        mesh.update()
        
        context.scene["taichi_hilbert_obj"] = obj.name
        context.scene.frame_set(context.scene.frame_start)
        run_hilbert_frame(context.scene)

        return {"FINISHED"}


class TAI_PT_hilbert_panel(bpy.types.Panel):
    bl_label = "Taichi Hilbert Physics"
    bl_idname = "TAI_PT_hilbert_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout = self.layout
        props = context.scene.taichi_hilbert

        box = layout.box()
        box.label(text="Generation Settings", icon="MODIFIER")
        col = box.column(align=True)
        col.prop(props, "order")
        col.prop(props, "size")
        col.prop(props, "drop_height")

        box = layout.box()
        box.label(text="Physics Settings", icon="PHYSICS")
        col = box.column(align=True)
        col.prop(props, "stiffness")
        col.prop(props, "damping")
        col.prop(props, "mass")
        col.prop(props, "dt")
        col.prop(props, "substeps")

        layout.separator()
        layout.operator(TAI_OT_create_hilbert.bl_idname, icon="CURVE_PATH", text="Create Simulation")

        # Compact Data Interface Disclosure
        layout.separator()
        layout.label(text="Data Interface", icon="SPREADSHEET")

        col = layout.column(align=True)
        row = col.row(align=True)
        box_name = row.box()
        box_name.scale_y = 0.5
        box_name.label(text="velocity_mag")
        box_domain = row.box()
        box_domain.scale_y = 0.5
        box_domain.label(text="Point")


classes = (
    TAI_HilbertProperties,
    TAI_OT_create_hilbert,
    TAI_PT_hilbert_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.taichi_hilbert = bpy.props.PointerProperty(type=TAI_HilbertProperties)

    # Surgical cleanup of existing handlers before registration
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre if h.__name__ != hilbert_update_handler.__name__
    ]
    bpy.app.handlers.frame_change_pre.append(hilbert_update_handler)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.taichi_hilbert

    if hilbert_update_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(hilbert_update_handler)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
