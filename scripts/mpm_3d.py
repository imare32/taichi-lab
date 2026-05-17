import bpy
import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# --- 1. MPM 核心常数与数据场 (3D) ---
quality = 1  # 提升此值以增加粒子数和网格精度
n_particles = 9000 * quality**3
n_grid = 32 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 3, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

# MPM 的粒子属性和网格属性 (3D)
x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)
material = ti.field(dtype=ti.i32, shape=n_particles)
Jp = ti.field(dtype=ti.f32, shape=n_particles)
grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
gravity = ti.Vector.field(3, dtype=ti.f32, shape=())


@ti.kernel
def init_mpm():
    """初始化粒子位置与材质 - 三个立方体 (居中分布)"""
    for i in range(n_particles):
        group_id = i // (n_particles // 3)
        # 调整初始位置，使其在 [0, 1]^3 空间中更居中且分布均匀
        x[i] = [
            ti.random() * 0.2 + 0.1 + 0.3 * group_id,  # 分布在 0.1-0.3, 0.4-0.6, 0.7-0.9
            ti.random() * 0.2 + 0.4,  # Y 轴居中 0.4-0.6
            ti.random() * 0.2 + 0.4 + 0.1 * (group_id % 2),  # Z 轴略微交错
        ]
        material[i] = group_id
        v[i] = [0, 0, 0]
        F[i] = ti.Matrix.identity(float, 3)
        Jp[i] = 1
        C[i] = ti.Matrix.zero(ti.f32, 3, 3)


@ti.kernel
def substep():
    """MPM 物理核心步长计算 (3D)"""
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0

    # 阶段 1: P2G (粒子到网格)
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # 3D 权重 (三次样条插值)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        # 变形梯度更新
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]

        # 材质模型 (流体, 果冻, 雪)
        h = ti.exp(10 * (1.0 - Jp[p]))
        if material[p] == 1:  # 果冻
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # 流体
            mu = 0.0

        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            if material[p] == 2:  # 雪
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig

        if material[p] == 0:
            F[p] = ti.Matrix.identity(float, 3) * ti.pow(J, 1 / 3)
        elif material[p] == 2:
            F[p] = U @ sig @ V.transpose()

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]

        # 3D 循环
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    # 阶段 2: 网格动量更新 & 边界碰撞
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]
            grid_v[i, j, k] += dt * gravity[None]
            # 3D 边界条件
            if i < 3 and grid_v[i, j, k][0] < 0:
                grid_v[i, j, k][0] = 0
            if i > n_grid - 3 and grid_v[i, j, k][0] > 0:
                grid_v[i, j, k][0] = 0
            if j < 3 and grid_v[i, j, k][1] < 0:
                grid_v[i, j, k][1] = 0
            if j > n_grid - 3 and grid_v[i, j, k][1] > 0:
                grid_v[i, j, k][1] = 0
            if k < 3 and grid_v[i, j, k][2] < 0:
                grid_v[i, j, k][2] = 0
            if k > n_grid - 3 and grid_v[i, j, k][2] > 0:
                grid_v[i, j, k][2] = 0

    # 阶段 3: G2P (网格到粒子)
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            dpos = ti.Vector([i, j, k]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] = ti.max(0.01, ti.min(0.99, x[p] + dt * v[p]))


# --- 2. Blender 交互与时间轴 ---
def run_simulation_frame_3d(scene):
    obj_name = scene.get("taichi_mpm_obj_3d", "")
    if not obj_name or obj_name not in bpy.data.objects:
        return

    # 从空物体获取 3D 重力
    gravity_obj_name = scene.get("taichi_mpm_gravity_obj_3d", "")
    fixed_mag = 50.0
    if gravity_obj_name in bpy.data.objects:
        empty = bpy.data.objects[gravity_obj_name]
        # 使用空物体的本地坐标作为重力方向
        loc = empty.location
        dist = loc.length
        if dist > 1e-5:
            gravity[None] = [
                loc.x / dist * fixed_mag,
                loc.y / dist * fixed_mag,
                loc.z / dist * fixed_mag,
            ]
        else:
            gravity[None] = [0.0, 0.0, -fixed_mag]
    else:
        gravity[None] = [0.0, 0.0, -fixed_mag]

    obj = bpy.data.objects[obj_name]
    mesh = obj.data

    if scene.frame_current <= scene.frame_start:
        init_mpm()
    else:
        for _ in range(25):  # 3D 计算较慢，稍微减少子步数
            substep()

    # 获取粒子的 3D 坐标
    taichi_pos = x.to_numpy()
    # 映射到 Blender 的 3D 空间 (10倍放大，并减去 0.5 偏移实现居中)
    np_pos = ((taichi_pos - 0.5) * 10.0).astype(np.float32)

    mesh.vertices.foreach_set("co", np_pos.flatten())
    mesh.update()
    ti.sync()


@bpy.app.handlers.persistent
def update_handler_3d(scene):
    run_simulation_frame_3d(scene)


# --- 3. Blender UI 操作符 ---
class TAI_OT_create_mpm_3d(bpy.types.Operator):
    bl_idname = "taichi.create_mpm_3d"
    bl_label = "Spawn MPM 3D Particles"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        init_mpm()
        verts = [(0, 0, 0)] * n_particles
        mesh = bpy.data.meshes.new("MPM_3D_Particles")
        mesh.from_pydata(verts, [], [])

        mat_np = material.to_numpy()
        attr = mesh.attributes.new(name="material_type", type="INT", domain="POINT")
        attr.data.foreach_set("value", mat_np)

        obj = bpy.data.objects.new("MPM_3D_Simulation", mesh)
        context.collection.objects.link(obj)
        context.scene["taichi_mpm_obj_3d"] = obj.name

        empty = bpy.data.objects.new("MPM_3D_Gravity_Control", None)
        empty.location = (0.0, 0.0, -5.0)
        context.collection.objects.link(empty)
        context.scene["taichi_mpm_gravity_obj_3d"] = empty.name

        context.scene.frame_set(1)
        run_simulation_frame_3d(context.scene)
        return {"FINISHED"}


class TAI_PT_mpm_panel_3d(bpy.types.Panel):
    bl_label = "Taichi MPM 3D Physics"
    bl_idname = "TAI_PT_mpm_panel_3d"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout = self.layout
        layout.operator(TAI_OT_create_mpm_3d.bl_idname, icon="PARTICLES")


def clear_handlers():
    bpy.app.handlers.frame_change_pre[:] = [h for h in bpy.app.handlers.frame_change_pre if h.__name__ != "update_handler_3d"]


# --- 4. 插件注册 ---
def register():
    bpy.utils.register_class(TAI_OT_create_mpm_3d)
    bpy.utils.register_class(TAI_PT_mpm_panel_3d)

    clear_handlers()
    if update_handler_3d not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(update_handler_3d)


def unregister():
    bpy.utils.unregister_class(TAI_PT_mpm_panel_3d)
    bpy.utils.unregister_class(TAI_OT_create_mpm_3d)

    clear_handlers()
    if update_handler_3d in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(update_handler_3d)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
