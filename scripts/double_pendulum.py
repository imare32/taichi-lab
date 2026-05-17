import bpy
import numpy as np
import taichi as ti
import math

# =========================
# 0. Safe Initialization
# =========================
if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu)
    except:
        ti.init(arch=ti.cpu)

# =========================
# 1. Physical Constants & Config (Internal Max)
# =========================
MAX_N = 64
MAX_POINTS = MAX_N * MAX_N
MAX_VERTS = MAX_POINTS * 6
EPS = 1e-2
OBJ_NAME = "PhaseSpace_Mesh"
MESH_NAME = "PhaseSpace_Data"

# =========================
# 2. Taichi Field Definitions
# =========================
th1 = ti.field(ti.f64, shape=MAX_POINTS)
th2 = ti.field(ti.f64, shape=MAX_POINTS)
dth1 = ti.field(ti.f64, shape=MAX_POINTS)
dth2 = ti.field(ti.f64, shape=MAX_POINTS)

verts_pos = ti.Vector.field(3, ti.f32, shape=MAX_VERTS)
verts_color = ti.Vector.field(4, ti.f32, shape=MAX_VERTS)

def hex_to_rgba(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return r, g, b, 1.0

C1_DATA = hex_to_rgba("#796BAF")
C2_DATA = hex_to_rgba("#719C73")

# =========================
# 3. Taichi Kernels
# =========================

@ti.func
def compute_accelerations(c_th1, c_th2, c_dth1, c_dth2, g, l1, l2, m1, m2):
    delta_th = c_th1 - c_th2
    sin_delta = ti.sin(delta_th)
    cos_delta = ti.cos(delta_th)
    mass_term = 2 * m1 + m2 - m2 * ti.cos(2 * delta_th)

    num1 = -g * (2 * m1 + m2) * ti.sin(c_th1)
    num2 = -m2 * g * ti.sin(c_th1 - 2 * c_th2)
    num3 = -2 * sin_delta * m2 * (c_dth2**2 * l2 + c_dth1**2 * l1 * cos_delta)
    ddth1 = (num1 + num2 + num3) / (l1 * mass_term)

    num4 = 2 * sin_delta
    num5 = c_dth1**2 * l1 * (m1 + m2)
    num6 = g * (m1 + m2) * ti.cos(c_th1)
    num7 = c_dth2**2 * l2 * m2 * cos_delta
    ddth2 = (num4 * (num5 + num6 + num7)) / (l2 * mass_term)
    return ddth1, ddth2

@ti.kernel
def init_phase_space(N: ti.i32):
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        θ1 = -math.pi + EPS + (i / (N - 1)) * (2 * math.pi - 2 * EPS)
        θ2 = -math.pi + EPS + (j / (N - 1)) * (2 * math.pi - 2 * EPS)
        th1[idx] = θ1
        th2[idx] = θ2
        dth1[idx] = 0.0
        dth2[idx] = 0.0

@ti.kernel
def step_rk4(dt: ti.f64, N: ti.i32, g: ti.f64, l1: ti.f64, l2: ti.f64, m1: ti.f64, m2: ti.f64):
    for idx in range(N * N):
        c_th1, c_th2 = th1[idx], th2[idx]
        c_dth1, c_dth2 = dth1[idx], dth2[idx]

        k1_dth1, k1_dth2 = compute_accelerations(c_th1, c_th2, c_dth1, c_dth2, g, l1, l2, m1, m2)
        k2_th1, k2_th2 = c_dth1 + 0.5 * dt * k1_dth1, c_dth2 + 0.5 * dt * k1_dth2
        k2_dth1, k2_dth2 = compute_accelerations(c_th1 + 0.5 * dt * c_dth1, c_th2 + 0.5 * dt * c_dth2, k2_th1, k2_th2, g, l1, l2, m1, m2)

        k3_th1, k3_th2 = c_dth1 + 0.5 * dt * k2_dth1, c_dth2 + 0.5 * dt * k2_dth2
        k3_dth1, k3_dth2 = compute_accelerations(c_th1 + 0.5 * dt * k2_th1, c_th2 + 0.5 * dt * k2_th2, k3_th1, k3_th2, g, l1, l2, m1, m2)

        k4_th1, k4_th2 = c_dth1 + dt * k3_dth1, c_dth2 + dt * k3_dth2
        k4_dth1, k4_dth2 = compute_accelerations(c_th1 + dt * k3_th1, c_th2 + dt * k3_th2, k4_th1, k4_th2, g, l1, l2, m1, m2)

        th1[idx] += (dt / 6.0) * (c_dth1 + 2 * k2_th1 + 2 * k3_th1 + k4_th1)
        th2[idx] += (dt / 6.0) * (c_dth2 + 2 * k2_th2 + 2 * k3_th2 + k4_th2)
        dth1[idx] += (dt / 6.0) * (k1_dth1 + 2 * k2_dth1 + 2 * k3_dth1 + k4_dth1)
        dth2[idx] += (dt / 6.0) * (k1_dth2 + 2 * k2_dth2 + 2 * k3_dth2 + k4_dth2)

@ti.kernel
def update_kinematics(N: ti.i32, l1: ti.f64, l2: ti.f64, spacing: ti.f64):
    c1 = ti.Vector(C1_DATA, dt=ti.f32)
    c2 = ti.Vector(C2_DATA, dt=ti.f32)
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        θ1, θ2 = th1[idx], th2[idx]
        x_base = (i - (N - 1) / 2.0) * spacing
        z_base = (j - (N - 1) / 2.0) * spacing

        p0 = ti.Vector([x_base, 0.0, z_base])
        p1 = p0 + ti.Vector([l1 * ti.sin(θ1), 0.0, -l1 * ti.cos(θ1)])
        p2 = p1 + ti.Vector([l2 * ti.sin(θ2), 0.0, -l2 * ti.cos(θ2)])

        v_idx = idx * 6
        verts_pos[v_idx + 0] = ti.cast(p0, ti.f32)
        verts_pos[v_idx + 1] = ti.cast(p1, ti.f32)
        verts_pos[v_idx + 2] = ti.cast(p1, ti.f32)
        verts_pos[v_idx + 3] = ti.cast(p2, ti.f32)
        verts_pos[v_idx + 4] = ti.cast(p1, ti.f32)
        verts_pos[v_idx + 5] = ti.cast(p1, ti.f32)

        verts_color[v_idx + 0] = c1
        verts_color[v_idx + 1] = c1
        verts_color[v_idx + 2] = c2
        verts_color[v_idx + 3] = c2
        verts_color[v_idx + 4] = c1
        verts_color[v_idx + 5] = c1

# =========================
# 4. Blender Integration Logic
# =========================

class TaichiPendulumProperties(bpy.types.PropertyGroup):
    n_grid: bpy.props.IntProperty(name="N (Grid Size)", default=20, min=2, max=MAX_N)
    gravity: bpy.props.FloatProperty(name="Gravity", default=9.8)
    l1: bpy.props.FloatProperty(name="Length 1", default=1.0)
    l2: bpy.props.FloatProperty(name="Length 2", default=1.0)
    m1: bpy.props.FloatProperty(name="Mass 1", default=1.0)
    m2: bpy.props.FloatProperty(name="Mass 2", default=1.0)
    spacing: bpy.props.FloatProperty(name="Spacing", default=2.2)

def setup_simulation_mesh(context, N):
    """Creates or updates the simulation mesh based on N."""
    scene = context.scene
    collection = context.collection

    props = scene.taichi_pendulum
    num_points = N * N
    num_verts = num_points * 6
    num_edges = num_points * 3

    # Taichi Logic
    init_phase_space(N)
    update_kinematics(N, props.l1, props.l2, props.spacing)

    # Mesh Logic
    mesh = bpy.data.meshes.new(MESH_NAME)
    mesh.vertices.add(num_verts)
    mesh.edges.add(num_edges)

    base_idx = np.arange(num_points, dtype=np.int32) * 6
    edges_array = np.column_stack([
        base_idx, base_idx + 1, 
        base_idx + 2, base_idx + 3, 
        base_idx + 4, base_idx + 5
    ])
    mesh.edges.foreach_set("vertices", edges_array.reshape(-1, 2).flatten())

    mesh.attributes.new(name="vertex_color", type="FLOAT_COLOR", domain="POINT")
    radius_attr = mesh.attributes.new(name="radius", type="FLOAT", domain="POINT")
    radius_pattern = np.tile([0.08, 0.08, 0.08, 0.08, 0.05, 0.05], num_points)
    radius_attr.data.foreach_set("value", radius_pattern)

    mesh.update()
    
    obj = bpy.data.objects.get(OBJ_NAME)
    if obj:
        old_mesh = obj.data
        obj.data = mesh
        if old_mesh and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    else:
        obj = bpy.data.objects.new(OBJ_NAME, mesh)
        collection.objects.link(obj)
    
    return obj

def run_pendulum_frame(scene):
    props = scene.taichi_pendulum
    obj = bpy.data.objects.get(OBJ_NAME)
    
    # Auto-initialize if at frame 1
    if scene.frame_current <= scene.frame_start:
        N_target = props.n_grid
        rebuild_needed = not obj
        if obj:
            actual_num_verts = len(obj.data.vertices)
            N_actual = int(math.sqrt(actual_num_verts / 6))
            if N_actual != N_target:
                rebuild_needed = True
        
        if rebuild_needed:
            # Use bpy.context when calling from a handler to fulfill 'context' requirement
            obj = setup_simulation_mesh(bpy.context, N_target)
        else:
            init_phase_space(N_target)
            
    if not obj or obj.mode != 'OBJECT':
        return

    mesh = obj.data
    actual_num_verts = len(mesh.vertices)
    N = int(math.sqrt(actual_num_verts / 6))
    
    if N * N * 6 != actual_num_verts:
        return

    if scene.frame_current > scene.frame_start:
        fps = scene.render.fps / scene.render.fps_base
        sub_steps = 4
        dt = (1.0 / fps) / sub_steps
        for _ in range(sub_steps):
            step_rk4(dt, N, props.gravity, props.l1, props.l2, props.m1, props.m2)

    update_kinematics(N, props.l1, props.l2, props.spacing)

    num_verts_curr = N * N * 6
    mesh.vertices.foreach_set("co", verts_pos.to_numpy()[:num_verts_curr].flatten())

    if "vertex_color" in mesh.attributes:
        mesh.attributes["vertex_color"].data.foreach_set("color", verts_color.to_numpy()[:num_verts_curr].flatten())

    mesh.update()
    ti.sync()

@bpy.app.handlers.persistent
def pendulum_update_handler(scene, depsgraph):
    run_pendulum_frame(scene)

class TAI_OT_create_pendulum(bpy.types.Operator):
    bl_idname = "taichi.create_pendulum"
    bl_label = "Initialize Simulation"
    bl_description = "Force initialize simulation and rebuild mesh"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        setup_simulation_mesh(context, context.scene.taichi_pendulum.n_grid)
        context.scene.frame_set(context.scene.frame_start)
        return {"FINISHED"}

class TAI_OT_select_target(bpy.types.Operator):
    bl_idname = "taichi.select_pendulum"
    bl_label = "Select Mesh"
    bl_description = "Select the generated simulation mesh"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = bpy.data.objects.get(OBJ_NAME)
        if obj:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
        return {'FINISHED'}

class TAI_PT_pendulum_panel(bpy.types.Panel):
    bl_label = "Taichi Double Pendulum"
    bl_idname = "TAI_PT_pendulum_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout = self.layout
        props = context.scene.taichi_pendulum
        obj = bpy.data.objects.get(OBJ_NAME)
        
        col = layout.column(align=True)
        col.label(text="Simulation Parameters:")
        col.prop(props, "n_grid")
        col.prop(props, "gravity")
        col.prop(props, "spacing")
        
        row = layout.row(align=True)
        row.prop(props, "l1")
        row.prop(props, "l2")
        
        row = layout.row(align=True)
        row.prop(props, "m1")
        row.prop(props, "m2")

        box = layout.box()
        box.operator(TAI_OT_create_pendulum.bl_idname, icon="PLAY")

        if obj:
            box.operator(TAI_OT_select_target.bl_idname, text=obj.name, icon="RESTRICT_SELECT_OFF")

            box = layout.box()
            box.label(text="Data Interface", icon='SPREADSHEET')
            row = box.row(align=True)
            row.label(text="Attribute")
            row.label(text="Domain")
            
            row = box.row(align=True)
            row.label(text="vertex_color")
            row.label(text="Point")
            
            row = box.row(align=True)
            row.label(text="radius")
            row.label(text="Point")

classes = (
    TaichiPendulumProperties,
    TAI_OT_create_pendulum,
    TAI_OT_select_target,
    TAI_PT_pendulum_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.taichi_pendulum = bpy.props.PointerProperty(type=TaichiPendulumProperties)

    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre 
        if h.__name__ != pendulum_update_handler.__name__
    ]
    bpy.app.handlers.frame_change_pre.append(pendulum_update_handler)

def unregister():
    if pendulum_update_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(pendulum_update_handler)
    del bpy.types.Scene.taichi_pendulum
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__" or __name__ == "<run_path>":
    register()
