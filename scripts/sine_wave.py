import bpy
import numpy as np
import taichi as ti

# --- 1. Taichi 初始化 ---
ti.init(arch=ti.gpu)


# --- 2. Taichi 动态 Kernel (核心魔法) ---
# 注意这里：我们不再使用 ti.field，而是用 ti.types.ndarray
# ndim=2 表示传入的是一个二维的 NumPy 数组，比如 shape 为 (N, 3) 的顶点数组
@ti.kernel
def compute_radial_sine_dynamic(
    verts: ti.types.ndarray(dtype=ti.f32, ndim=2),  # 输出：计算后的 3D 顶点 (N, 3)
    xy: ti.types.ndarray(dtype=ti.f32, ndim=2),  # 输入：原始的 2D 坐标 (N, 2)
    phase: ti.f32,
):
    # verts.shape[0] 会在运行时动态获取传入的 NumPy 数组的长度！不会写死！
    for i in range(verts.shape[0]):
        x = xy[i, 0]
        y = xy[i, 1]

        dist = ti.math.sqrt(x * x + y * y)
        z = ti.math.sin(dist * 2.0 - phase) * 0.5

        verts[i, 0] = x
        verts[i, 1] = y
        verts[i, 2] = z


# --- 3. 纯 Python/NumPy 的辅助生成函数 ---
def generate_grid_data(res, size=5.0):
    """根据分辨率高速生成网格的初始 XY 坐标和面索引"""
    step = size / (res - 1)
    offset = size / 2.0

    # 极速生成 XY 坐标矩阵
    x = np.linspace(-offset, offset, res, dtype=np.float32)
    y = np.linspace(-offset, offset, res, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    # 展平为 (N, 2) 的形状供 Taichi 读取
    xy_arr = np.empty((res * res, 2), dtype=np.float32)
    xy_arr[:, 0] = xv.flatten()
    xy_arr[:, 1] = yv.flatten()

    # 生成四边形面拓扑
    faces = []
    for i in range(res - 1):
        for j in range(res - 1):
            v0 = i * res + j
            v1 = v0 + 1
            v2 = v0 + res + 1
            v3 = v0 + res
            faces.append((v0, v1, v2, v3))

    return xy_arr, faces


# --- 4. 核心更新逻辑 (响应 UI 滑块) ---
def update_taichi_mesh(self, context):
    obj_name = context.scene.get("taichi_target_obj", "")
    if not obj_name or obj_name not in bpy.data.objects:
        return

    obj = bpy.data.objects[obj_name]
    mesh = obj.data

    res = context.scene.taichi_res
    phase = context.scene.taichi_phase
    vertex_count = res * res

    # 1. 检查是否需要重建拓扑 (如果滑动了分辨率滑块，顶点数会变)
    if len(mesh.vertices) != vertex_count:
        xy_arr, faces = generate_grid_data(res)
        # 准备一个全零的 3D 数组用于初始化网格结构
        verts_3d_init = np.zeros((vertex_count, 3), dtype=np.float32)
        verts_3d_init[:, :2] = xy_arr

        # 清空旧网格并灌入新拓扑结构 (Blender 原生操作)
        mesh.clear_geometry()
        mesh.from_pydata(verts_3d_init.tolist(), [], faces)
    else:
        # 如果分辨率没变 (只是在拖动相位)，只提取 XY 坐标即可
        xy_arr, _ = generate_grid_data(res)

    # 2. 创建动态的 NumPy 数组作为 Taichi 的输出容器
    verts_arr = np.zeros((vertex_count, 3), dtype=np.float32)

    # 3. ⚡ 呼叫 Taichi (无缝传入 NumPy 数组，零拷贝传递指针)
    compute_radial_sine_dynamic(verts_arr, xy_arr, phase)

    # 4. 将 Taichi 算好的数据展平，极速打回 Blender
    mesh.vertices.foreach_set("co", verts_arr.flatten())
    mesh.update()


# --- 5. Blender 操作符与 UI 面板 ---
class TAI_OT_create_grid(bpy.types.Operator):
    """创建并绑定 Taichi 驱动的网格"""

    bl_idname = "taichi.create_grid"
    bl_label = "Create Dynamic Grid"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mesh = bpy.data.meshes.new("TaichiWaveMesh")
        obj = bpy.data.objects.new("TaichiWaveGrid", mesh)
        context.collection.objects.link(obj)
        context.scene["taichi_target_obj"] = obj.name

        # 强制更新一次以生成初始网格
        update_taichi_mesh(self, context)
        return {"FINISHED"}


class TAI_PT_main_panel(bpy.types.Panel):
    bl_label = "Taichi Generator"
    bl_idname = "TAI_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.operator(TAI_OT_create_grid.bl_idname, icon="MESH_GRID")
        layout.separator()

        # UI 现在有了两个滑块：分辨率 和 相位
        layout.prop(scene, "taichi_res")
        layout.prop(scene, "taichi_phase")


# --- 6. 属性注册 ---
def register():
    bpy.utils.register_class(TAI_OT_create_grid)
    bpy.utils.register_class(TAI_PT_main_panel)

    # 注册分辨率属性 (整数，最小 2x2，最大 200x200)
    bpy.types.Scene.taichi_res = bpy.props.IntProperty(
        name="Resolution",
        description="Grid resolution (NxN)",
        default=50,  # 默认 50x50 = 2500 个顶点
        min=2,
        max=200,
        update=update_taichi_mesh,
    )

    bpy.types.Scene.taichi_phase = bpy.props.FloatProperty(name="Wave Phase", default=0.0, update=update_taichi_mesh)


def unregister():
    bpy.utils.unregister_class(TAI_OT_create_grid)
    bpy.utils.unregister_class(TAI_PT_main_panel)

    del bpy.types.Scene.taichi_res
    del bpy.types.Scene.taichi_phase


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
