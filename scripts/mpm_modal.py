import blf
import bpy
import gpu
import numpy as np
import taichi as ti
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader

# --- 1. MPM 核心逻辑 ---
ti.init(arch=ti.gpu)

quality = 1
n_particles = 9000 * quality**2
n_grid = 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

x = ti.Vector.field(2, dtype=float, shape=n_particles)
v = ti.Vector.field(2, dtype=float, shape=n_particles)
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
material = ti.field(dtype=int, shape=n_particles)
Jp = ti.field(dtype=float, shape=n_particles)
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))
gravity = ti.Vector.field(2, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())


@ti.kernel
def reset_mpm_kernel():
    group_size = n_particles // 3
    for i in range(n_particles):
        x[i] = [ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size), ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)]
        material[i] = i // group_size
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)
    gravity[None] = [0, -1]
    attractor_strength[None] = 0


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))
        if material[p] == 1:
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0:
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 2:
            F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 1e-10:
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j] += dt * gravity[None] * 30
            dist = attractor_pos[None] - dx * ti.Vector([i, j])
            grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]


# --- 2. 仿真 Handler ---


def update_handler(scene):
    obj_name = scene.get("taichi_modal_obj", "")
    if not obj_name or obj_name not in bpy.data.objects:
        return
    obj = bpy.data.objects[obj_name]

    if scene.frame_current <= scene.frame_start:
        reset_mpm_kernel()
    else:
        for _ in range(25):
            substep()

    taichi_pos = x.to_numpy()
    np_pos = np.zeros((n_particles, 3), dtype=np.float32)
    np_pos[:, 0] = (taichi_pos[:, 0] - 0.5) * 10.0
    np_pos[:, 2] = (taichi_pos[:, 1] - 0.5) * 10.0
    obj.data.vertices.foreach_set("co", np_pos.flatten())
    obj.data.update()


# --- 3. Modal 交互 & 绘制 ---


def draw_callback_2d(self, context):
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    width = context.region.width
    height = context.region.height
    padding = 2
    coords = [(padding, padding), (width - padding, padding), (width - padding, height - padding), (padding, height - padding)]
    indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
    batch = batch_for_shader(shader, "LINES", {"pos": coords}, indices=indices)
    shader.bind()
    shader.uniform_float("color", (1, 0, 0, 1))
    gpu.state.line_width_set(2.0)
    batch.draw(shader)

    # UI Feedback for WASD
    font_id = 0
    blf.size(font_id, 24)
    blf.color(font_id, 1.0, 0.8, 0.2, 1.0)  # Yellowish

    keys = getattr(self, "keys", {})
    active_keys = [k for k in ["W", "A", "S", "D"] if keys.get(k)]

    text = "Gravity: "
    if not active_keys:
        text += "DOWN (Default)"
    else:
        dirs = []
        if "W" in active_keys:
            dirs.append("UP")
        if "S" in active_keys:
            dirs.append("DOWN")
        if "A" in active_keys:
            dirs.append("LEFT")
        if "D" in active_keys:
            dirs.append("RIGHT")
        text += " + ".join(dirs)

    blf.position(font_id, 20, 40, 0)
    blf.draw(font_id, text)


class TAI_OT_mpm_modal(bpy.types.Operator):
    bl_idname = "taichi.mpm_modal"
    bl_label = "Interactive Modal Internal"
    bl_options = {"REGISTER"}

    _handle = None

    def modal(self, context, event):
        if context.area:
            context.area.tag_redraw()

        # 核心：检查 Scene 属性，如果被 UI 或 ESC 设为 False，则退出
        if not context.scene.taichi_mpm_is_active:
            self.remove_handle(context)
            return {"CANCELLED"}

        if event.type == "ESC" and event.value == "PRESS":
            context.scene.taichi_mpm_is_active = False  # 触发 update 回调
            return {"RUNNING_MODAL"}

        # 拦截并处理 WASD 控制重力方向
        if event.type in {"W", "A", "S", "D"}:
            if event.value in {"PRESS", "RELEASE"}:
                self.keys[event.type] = event.value == "PRESS"
                gx, gy = 0.0, 0.0
                if self.keys.get("A"):
                    gx -= 1.0
                if self.keys.get("D"):
                    gx += 1.0
                if self.keys.get("W"):
                    gy += 1.0
                if self.keys.get("S"):
                    gy -= 1.0
                if gx == 0.0 and gy == 0.0:
                    gy = -1.0  # 默认重力
                gravity[None] = [gx, gy]
            return {"RUNNING_MODAL"}  # 核心：阻止这些按键触发 Blender 默认功能

        # 投影与交互逻辑
        window_region = next((r for r in context.area.regions if r.type == "WINDOW"), None)
        rv3d = context.space_data.region_3d if hasattr(context, "space_data") else None

        if window_region and rv3d:
            # 使用基于整个 Blender 窗口的屏幕坐标转换为 3D View 的相对坐标
            mx = event.mouse_x - window_region.x
            my = event.mouse_y - window_region.y
            coord = (mx, my)

            ray_origin = view3d_utils.region_2d_to_origin_3d(window_region, rv3d, coord)
            ray_direction = view3d_utils.region_2d_to_vector_3d(window_region, rv3d, coord)
            if abs(ray_direction.y) > 1e-6:
                t = -ray_origin.y / ray_direction.y
                hit_pos = ray_origin + t * ray_direction
                attractor_pos[None] = [(hit_pos.x / 10.0) + 0.5, (hit_pos.z / 10.0) + 0.5]

        if event.type == "LEFTMOUSE":
            if event.value == "PRESS":
                attractor_strength[None] = 1.0
            elif event.value == "RELEASE":
                attractor_strength[None] = 0.0
            return {"PASS_THROUGH"}

        elif event.type == "RIGHTMOUSE":
            if event.value == "PRESS":
                attractor_strength[None] = -1.0
                return {"RUNNING_MODAL"}  # 仅拦截右键按下，阻止菜单弹出
            elif event.value == "RELEASE":
                attractor_strength[None] = 0.0
                return {"PASS_THROUGH"}

        return {"PASS_THROUGH"}

    def invoke(self, context, event):
        if context.area.type == "VIEW_3D":
            if TAI_OT_mpm_modal._handle:
                return {"CANCELLED"}
            self.keys = {"W": False, "A": False, "S": False, "D": False}
            self.__class__._handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_2d, (self, context), "WINDOW", "POST_PIXEL")
            context.window_manager.modal_handler_add(self)

            obj_name = context.scene.get("taichi_modal_obj", "")
            if obj_name in bpy.data.objects:
                obj = bpy.data.objects[obj_name]
                obj.hide_select = True
                obj.select_set(False)

            return {"RUNNING_MODAL"}
        return {"CANCELLED"}

    def remove_handle(self, context):
        if self.__class__._handle:
            bpy.types.SpaceView3D.draw_handler_remove(self.__class__._handle, "WINDOW")
            self.__class__._handle = None
        attractor_strength[None] = 0.0
        gravity[None] = [0, -1]  # 恢复默认重力

        obj_name = context.scene.get("taichi_modal_obj", "")
        if obj_name in bpy.data.objects:
            bpy.data.objects[obj_name].hide_select = False

        obj_name = context.scene.get("taichi_modal_obj", "")
        if obj_name in bpy.data.objects:
            bpy.data.objects[obj_name].hide_select = False


def update_mpm_active(self, context):
    if self.taichi_mpm_is_active:
        if TAI_OT_mpm_modal._handle is None:
            # 必须使用 INVOKE_DEFAULT 来启动 Modal
            bpy.ops.taichi.mpm_modal("INVOKE_DEFAULT")


# --- 4. 初始化 & 面板 ---


class TAI_OT_mpm_init(bpy.types.Operator):
    bl_idname = "taichi.mpm_init"
    bl_label = "Initialize MPM"

    def execute(self, context):
        reset_mpm_kernel()
        obj_name = "MPM_Sim_Object"
        if obj_name not in bpy.data.objects:
            mesh = bpy.data.meshes.new(obj_name + "_Mesh")
            obj = bpy.data.objects.new(obj_name, mesh)
            context.collection.objects.link(obj)
            mesh.from_pydata([(0, 0, 0)] * n_particles, [], [])

            mat_np = material.to_numpy().astype(np.float32)
            mat_np /= 3.0  # 归一化材质
            attr = mesh.attributes.new(name="material_type", type="FLOAT", domain="POINT")
            attr.data.foreach_set("value", mat_np)
        else:
            obj = bpy.data.objects[obj_name]

        taichi_pos = x.to_numpy()
        np_pos = np.zeros((n_particles, 3), dtype=np.float32)
        np_pos[:, 0] = (taichi_pos[:, 0] - 0.5) * 10.0
        np_pos[:, 2] = (taichi_pos[:, 1] - 0.5) * 10.0
        obj.data.vertices.foreach_set("co", np_pos.flatten())
        obj.data.update()
        context.scene["taichi_modal_obj"] = obj_name
        return {"FINISHED"}


class TAI_PT_mpm_panel(bpy.types.Panel):
    bl_label = "Taichi MPM Interaction"
    bl_idname = "TAI_PT_mpm_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.operator("taichi.mpm_init", icon="PARTICLES")

        # 核心：使用 BoolProperty 切换，它会自动调用 update 回调
        layout.prop(scene, "taichi_mpm_is_active", text="Interaction Mode", toggle=True, icon="MOUSE_MOVE")

        layout.separator()
        layout.label(text="LMB: Attract | RMB: Repel")
        layout.label(text="WASD: Control Gravity")
        layout.label(text="ESC: Exit Modal")


# --- 5. 注册与清理 ---


def clear_handlers():
    bpy.app.handlers.frame_change_pre[:] = [h for h in bpy.app.handlers.frame_change_pre if h.__name__ != "update_handler"]


def register():
    bpy.utils.register_class(TAI_OT_mpm_init)
    bpy.utils.register_class(TAI_OT_mpm_modal)
    bpy.utils.register_class(TAI_PT_mpm_panel)

    bpy.types.Scene.taichi_mpm_is_active = bpy.props.BoolProperty(name="Interactive Mode", description="Toggle real-time mouse interaction", default=False, update=update_mpm_active)

    clear_handlers()
    bpy.app.handlers.frame_change_pre.append(update_handler)


def unregister():
    clear_handlers()
    if hasattr(bpy.types.Scene, "taichi_mpm_is_active"):
        del bpy.types.Scene.taichi_mpm_is_active
    bpy.utils.unregister_class(TAI_PT_mpm_panel)
    bpy.utils.unregister_class(TAI_OT_mpm_modal)
    bpy.utils.unregister_class(TAI_OT_mpm_init)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
