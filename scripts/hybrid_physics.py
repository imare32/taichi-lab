import bpy
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)


# =========================
# 1. 物理参数与配置
# =========================
N = 128
WAVE_DT = 0.1
HEAT_DT = 0.02
WAVE_SPEED = 0.4  # 提高波速，让反弹更明显
DAMPING = 0.9985  # 显著减小阻尼，让波纹更持久

# =========================
# 2. Taichi 数据场
# =========================
pos = ti.Vector.field(3, ti.f32, shape=(N, N))
vel_z = ti.field(ti.f32, shape=(N, N))
buffer_z = ti.field(ti.f32, shape=(N, N))


@ti.kernel
def init_sim():
    for i, j in pos:
        pos[i, j] = ti.Vector([(i - N / 2) * 0.1, (j - N / 2) * 0.1, 0.0])
        vel_z[i, j] = 0.0
        buffer_z[i, j] = 0.0


@ti.kernel
def update_physics(do_wave: bool, do_heat: bool):
    # 1. 计算扩散 (针对 pos.z)
    for i, j in ti.ndrange((1, N - 1), (1, N - 1)):
        laplace = pos[i - 1, j].z + pos[i + 1, j].z + pos[i, j - 1].z + pos[i, j + 1].z - pos[i, j].z * 4

        target_z = pos[i, j].z
        if do_heat:
            target_z += laplace * HEAT_DT

        if do_wave:
            # 波动更新：加速度由拉普拉斯算子决定
            vel_z[i, j] += laplace * WAVE_SPEED * WAVE_DT
            vel_z[i, j] *= DAMPING  # 阻尼
            target_z += vel_z[i, j] * WAVE_DT

        buffer_z[i, j] = target_z

    # 2. 同步回主场
    for i, j in ti.ndrange((1, N - 1), (1, N - 1)):
        pos[i, j].z = buffer_z[i, j]


@ti.kernel
def drop_impact(x: int, y: int, strength: float, radius: float):
    for i, j in pos:
        if i > 0 and i < N - 1 and j > 0 and j < N - 1:
            dist_sqr = float((x - i) ** 2 + (y - j) ** 2)
            pos[i, j].z += strength * ti.exp(-radius * dist_sqr)


# =========================
# 3. Blender 交互逻辑
# =========================


def run_hybrid_frame(scene):
    obj_name = scene.get("taichi_hybrid_obj", "")
    if not obj_name or obj_name not in bpy.data.objects:
        return

    obj = bpy.data.objects[obj_name]
    mesh = obj.data

    if scene.frame_current <= scene.frame_start:
        init_sim()
    else:
        # 自动投放
        if scene.taichi_auto_drop:
            if scene.frame_current % 20 == 0:
                x, y = np.random.randint(1, N - 1, 2)
                # 随机选择投放水滴还是岩浆
                if np.random.rand() > 0.5:
                    drop_impact(x, y, 1.5, 0.08)  # 水滴
                else:
                    drop_impact(x, y, 4.0, 0.02)  # 岩浆

        # 物理步进
        for _ in range(8):
            update_physics(scene.taichi_do_wave, scene.taichi_do_heat)

    # 极速同步数据
    mesh.vertices.foreach_set("co", pos.to_numpy().flatten())
    mesh.update()
    ti.sync()


@bpy.app.handlers.persistent
def hybrid_update_handler(scene):
    run_hybrid_frame(scene)


class TAI_OT_create_hybrid(bpy.types.Operator):
    bl_idname = "taichi.create_hybrid"
    bl_label = "Spawn Hybrid Fluid/Thermal Mesh"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        init_sim()

        # 创建网格拓扑
        res = N
        size = 0.1
        verts = []
        faces = []
        for i in range(res):
            for j in range(res):
                verts.append(((i - res / 2) * size, (j - res / 2) * size, 0))

        for i in range(res - 1):
            for j in range(res - 1):
                v0 = i * res + j
                v1 = v0 + 1
                v2 = (i + 1) * res + j + 1
                v3 = (i + 1) * res + j
                faces.append((v0, v1, v2, v3))

        mesh = bpy.data.meshes.new("Hybrid_Physics_Mesh")
        mesh.from_pydata(verts, [], faces)
        mesh.update()

        obj = bpy.data.objects.new("Hybrid_Simulation", mesh)
        context.collection.objects.link(obj)
        context.scene["taichi_hybrid_obj"] = obj.name

        context.scene.frame_set(context.scene.frame_start)
        return {"FINISHED"}


class TAI_OT_drop_lava(bpy.types.Operator):
    bl_idname = "taichi.drop_lava"
    bl_label = "Drop Lava"

    def execute(self, context):
        x, y = np.random.randint(10, N - 10, 2)
        drop_impact(int(x), int(y), 5.0, 0.02)
        return {"FINISHED"}


class TAI_OT_drop_water(bpy.types.Operator):
    bl_idname = "taichi.drop_water"
    bl_label = "Drop Water"

    def execute(self, context):
        x, y = np.random.randint(10, N - 10, 2)
        drop_impact(int(x), int(y), 2.0, 0.1)
        return {"FINISHED"}


class TAI_PT_hybrid_panel(bpy.types.Panel):
    bl_label = "Taichi Hybrid Physics"
    bl_idname = "TAI_PT_hybrid_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.operator(TAI_OT_create_hybrid.bl_idname, icon="GRID")

        if scene.get("taichi_hybrid_obj", "") in bpy.data.objects:
            box = layout.box()
            box.label(text="Manual Interaction")
            row = box.row(align=True)
            row.operator(TAI_OT_drop_lava.bl_idname, icon="COLORSET_01_VEC")
            row.operator(TAI_OT_drop_water.bl_idname, icon="COLORSET_04_VEC")

            box = layout.box()
            box.label(text="Simulation Settings")
            box.prop(scene, "taichi_auto_drop", text="Auto Random Drop")
            box.prop(scene, "taichi_do_wave", text="Enable Wave (Ripples)")
            box.prop(scene, "taichi_do_heat", text="Enable Heat (Diffusion)")


def register():
    bpy.utils.register_class(TAI_OT_create_hybrid)
    bpy.utils.register_class(TAI_OT_drop_lava)
    bpy.utils.register_class(TAI_OT_drop_water)
    bpy.utils.register_class(TAI_PT_hybrid_panel)

    bpy.types.Scene.taichi_auto_drop = bpy.props.BoolProperty(name="Auto Drop", default=True)
    bpy.types.Scene.taichi_do_wave = bpy.props.BoolProperty(name="Do Wave", default=True)
    bpy.types.Scene.taichi_do_heat = bpy.props.BoolProperty(name="Do Heat", default=True)
    if hybrid_update_handler not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(hybrid_update_handler)


def unregister():
    bpy.utils.unregister_class(TAI_PT_hybrid_panel)
    bpy.utils.unregister_class(TAI_OT_drop_water)
    bpy.utils.unregister_class(TAI_OT_drop_lava)
    bpy.utils.unregister_class(TAI_OT_create_hybrid)

    del bpy.types.Scene.taichi_auto_drop
    del bpy.types.Scene.taichi_do_wave
    del bpy.types.Scene.taichi_do_heat
    if hybrid_update_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(hybrid_update_handler)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
