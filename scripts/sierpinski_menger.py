import bpy
import numpy as np
import taichi as ti

bl_info = {
    "name": "Taichi Fractal GN Lab Pro",
    "author": "Gemini Assistant",
    "version": (4, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Taichi Tab",
    "description": "Taichi point clouds with automatic Geometry Nodes and Attributes",
    "category": "Mesh",
}

# --- Taichi Setup ---
try:
    ti.init(arch=ti.gpu, log_level=ti.INFO)
except:
    ti.init(arch=ti.cpu)

menger_2d_lut = ti.field(dtype=ti.i32, shape=(8, 2))
menger_3d_lut = ti.field(dtype=ti.i32, shape=(20, 3))


def init_luts():
    # 2D: 将 y 设为外层循环，使拓扑索引优先沿 Y 轴（垂直）增长
    l2 = [[x, y] for y in range(3) for x in range(3) if not (x == 1 and y == 1)]
    for i in range(8):
        menger_2d_lut[i, 0], menger_2d_lut[i, 1] = l2[i][0], l2[i][1]

    # 3D: 将 z 设为最外层循环，使大尺度递归块优先沿 Z 轴堆叠
    l3 = [[x, y, z] for z in range(3) for y in range(3) for x in range(3) if (x == 1) + (y == 1) + (z == 1) < 2]
    for i in range(20):
        menger_3d_lut[i, 0], menger_3d_lut[i, 1], menger_3d_lut[i, 2] = l3[i][0], l3[i][1], l3[i][2]


# --- Point Generation Kernels ---


@ti.kernel
def gen_points_menger_3d(level: ti.i32, size: ti.f32, out_points: ti.types.ndarray(), out_depths: ti.types.ndarray(), out_iters: ti.types.ndarray()):
    num = 20**level
    side = 3**level
    cs = size / side
    for i in range(num):
        ix, iy, iz = 0, 0, 0
        temp = i
        for l in range(level):
            choice = temp % 20
            temp //= 20
            scale = 3**l
            ix += menger_3d_lut[choice, 0] * scale
            iy += menger_3d_lut[choice, 1] * scale
            iz += menger_3d_lut[choice, 2] * scale
        out_points[i, 0] = (ix - side / 2 + 0.5) * cs
        out_points[i, 1] = (iy - side / 2 + 0.5) * cs
        out_points[i, 2] = (iz - side / 2 + 0.5) * cs + size / 2
        out_depths[i] = i / ti.f32(num)  # 还原拓扑逻辑
        out_iters[i] = (i % 20) / 19.0


@ti.kernel
def gen_points_menger_2d(level: ti.i32, size: ti.f32, out_points: ti.types.ndarray(), out_depths: ti.types.ndarray(), out_iters: ti.types.ndarray()):
    num = 8**level
    side = 3**level
    cs = size / side
    for i in range(num):
        ix, iy = 0, 0
        temp = i
        for l in range(level):
            choice = temp % 8
            temp //= 8
            scale = 3**l
            ix += menger_2d_lut[choice, 0] * scale
            iy += menger_2d_lut[choice, 1] * scale
        out_points[i, 0], out_points[i, 1], out_points[i, 2] = (ix - side / 2 + 0.5) * cs, (iy - side / 2 + 0.5) * cs, 0.0
        out_depths[i] = i / ti.f32(num)  # 还原拓扑逻辑
        out_iters[i] = (i % 8) / 7.0


@ti.kernel
def gen_points_sierpinski_3d(level: ti.i32, size: ti.f32, out_points: ti.types.ndarray(), out_depths: ti.types.ndarray(), out_iters: ti.types.ndarray()):
    num = 4**level
    h, H = size * 0.866025, size * 0.816496
    v_init = [ti.Vector([-size / 2, -h / 3, 0.0]), ti.Vector([size / 2, -h / 3, 0.0]), ti.Vector([0.0, 2 * h / 3, 0.0]), ti.Vector([0.0, 0.0, H])]
    for i in range(num):
        v = [v_init[0], v_init[1], v_init[2], v_init[3]]
        temp = i
        for _ in range(level):
            c = temp % 4
            temp //= 4
            m01, m02, m03 = (v[0] + v[1]) * 0.5, (v[0] + v[2]) * 0.5, (v[0] + v[3]) * 0.5
            m12, m13, m23 = (v[1] + v[2]) * 0.5, (v[1] + v[3]) * 0.5, (v[2] + v[3]) * 0.5
            if c == 0:
                v[1], v[2], v[3] = m01, m02, m03
            elif c == 1:
                v[0], v[2], v[3] = m01, m12, m13
            elif c == 2:
                v[0], v[1], v[3] = m02, m12, m23
            else:
                v[0], v[1], v[2] = m03, m13, m23
        p = (v[0] + v[1] + v[2] + v[3]) * 0.25
        out_points[i, 0], out_points[i, 1], out_points[i, 2] = p.x, p.y, p.z
        out_depths[i] = i / ti.f32(num)  # 拓扑深度（Sierpinski 顶点 v3 本身就在顶部，无需额外调整）
        out_iters[i] = (i % 4) / 3.0


@ti.kernel
def gen_points_sierpinski_2d(level: ti.i32, size: ti.f32, out_points: ti.types.ndarray(), out_depths: ti.types.ndarray(), out_iters: ti.types.ndarray()):
    num = 3**level
    h = size * 0.866025
    v_init = [ti.Vector([-size / 2, -h / 3, 0.0]), ti.Vector([size / 2, -h / 3, 0.0]), ti.Vector([0.0, 2 * h / 3, 0.0])]
    for i in range(num):
        v = [v_init[0], v_init[1], v_init[2]]
        temp = i
        for _ in range(level):
            c = temp % 3
            temp //= 3
            m01, m12, m20 = (v[0] + v[1]) * 0.5, (v[1] + v[2]) * 0.5, (v[2] + v[0]) * 0.5
            if c == 0:
                v[1], v[2] = m01, m20
            elif c == 1:
                v[0], v[2] = m01, m12
            else:
                v[0], v[1] = m20, m12
        p = (v[0] + v[1] + v[2]) / 3.0
        out_points[i, 0], out_points[i, 1], out_points[i, 2] = p.x, p.y, 0.0
        out_depths[i] = i / ti.f32(num)  # 拓扑深度
        out_iters[i] = (i % 3) / 2.0


# --- Blender Logic & GN Setup ---


def setup_fractal_instances_gn_robust(cloud_obj, base_obj):
    group_name = "Fractal Instances"
    node_group = bpy.data.node_groups.get(group_name)

    # 核心节点检查机制，防止用户误删导致脚本挂掉
    needs_rebuild = True
    if node_group:
        node_types = [n.type for n in node_group.nodes]
        if "OBJECT_INFO" in node_types and "INSTANCE_ON_POINTS" in node_types:
            needs_rebuild = False

    if needs_rebuild:
        if node_group:
            bpy.data.node_groups.remove(node_group)
        node_group = bpy.data.node_groups.new(type="GeometryNodeTree", name=group_name)

        if hasattr(node_group, "interface"):
            node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        else:
            node_group.inputs.new("NodeSocketGeometry", "Geometry")
            node_group.outputs.new("NodeSocketGeometry", "Geometry")

        nodes = node_group.nodes
        links = node_group.links

        group_input = nodes.new("NodeGroupInput")
        group_output = nodes.new("NodeGroupOutput")
        object_info = nodes.new("GeometryNodeObjectInfo")
        instance_on_points = nodes.new("GeometryNodeInstanceOnPoints")

        object_info.transform_space = "ORIGINAL"

        links.new(group_input.outputs[0], instance_on_points.inputs[0])
        links.new(object_info.outputs[4], instance_on_points.inputs[2])  # Geometry to Instance
        links.new(instance_on_points.outputs[0], group_output.inputs[0])

        group_input.location = (-200, 160)
        object_info.location = (-200, 60)
        instance_on_points.location = (30, 160)
        group_output.location = (220, 130)

    for node in node_group.nodes:
        if node.type == "OBJECT_INFO":
            node.inputs[0].default_value = base_obj
            node.inputs[1].default_value = True  # 强制 As Instance，确保着色器能继承 Instancer 属性
            break

    mod = cloud_obj.modifiers.get("FractalInstancer")
    if not mod:
        mod = cloud_obj.modifiers.new("FractalInstancer", "NODES")
    mod.node_group = node_group


def update_base_geometry(mesh, f_type, mode, scale):
    mesh.clear_geometry()
    s = scale / 2.0
    if f_type == "MENGER":
        if mode == "3D":
            verts = [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s), (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]
            mesh.from_pydata(verts, [], [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
        else:
            mesh.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    else:
        h = scale * 0.866025
        if mode == "3D":
            H = scale * 0.816496
            z_bot = -H / 4.0
            z_top = 3.0 * H / 4.0
            verts = [(-s, -h / 3, z_bot), (s, -h / 3, z_bot), (0, 2 * h / 3, z_bot), (0, 0, z_top)]
            mesh.from_pydata(verts, [], [(2, 1, 0), (0, 1, 3), (1, 2, 3), (2, 0, 3)])
        else:
            mesh.from_pydata([(-s, -h / 3, 0), (s, -h / 3, 0), (0, 2 * h / 3, 0)], [], [(0, 1, 2)])


# 内存与显存溢出的刚性保护红线
MAX_SAFE_POINTS = 3000000


def show_message_box(message="", title="Taichi Lab Warning", icon="INFO"):
    def draw(self, context):
        self.layout.label(text=message)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def update_mesh(self, context):
    props = context.scene.taichi_fractal_props
    f_type, mode, level, size = props.fractal_type, props.mode, props.level, props.size

    # 拦截计算，防止崩溃
    if f_type == "MENGER":
        count = 20**level if mode == "3D" else 8**level
    else:
        count = 4**level if mode == "3D" else 3**level

    if count > MAX_SAFE_POINTS:
        msg = f"Level {level} generates {count} points. Blocked to prevent OOM!"
        show_message_box(msg, icon="ERROR")
        return

    cloud_name = "Taichi_Cloud"
    base_name = "Taichi_Mesh"

    # 强制获取或重建目标物体
    cloud_obj = bpy.data.objects.get(cloud_name)
    if not cloud_obj or cloud_obj.type != "MESH":
        mesh = bpy.data.meshes.new(cloud_name + "_Mesh")
        cloud_obj = bpy.data.objects.new(cloud_name, mesh)
        context.scene.collection.objects.link(cloud_obj)

    base_obj = bpy.data.objects.get(base_name)
    if not base_obj or base_obj.type != "MESH":
        mesh = bpy.data.meshes.new(base_name + "_Mesh")
        base_obj = bpy.data.objects.new(base_name, mesh)
        context.scene.collection.objects.link(base_obj)
        base_obj.hide_set(True)

    pts = np.zeros((count, 3), dtype=np.float32)
    depths = np.zeros(count, dtype=np.float32)
    iters = np.zeros(count, dtype=np.float32)

    unit_scale = 1.0
    if f_type == "MENGER":
        unit_scale = size / (3**level)
        if mode == "3D":
            gen_points_menger_3d(level, size, pts, depths, iters)
        else:
            gen_points_menger_2d(level, size, pts, depths, iters)
    else:
        unit_scale = size / (2**level)
        if mode == "3D":
            gen_points_sierpinski_3d(level, size, pts, depths, iters)
        else:
            gen_points_sierpinski_2d(level, size, pts, depths, iters)

    # 写入顶点位置
    cloud_mesh = cloud_obj.data
    cloud_mesh.clear_geometry()
    cloud_mesh.vertices.add(count)
    cloud_mesh.vertices.foreach_set("co", pts.ravel())

    # 写入自定义属性 Fractal_Depth
    depth_attr = cloud_mesh.attributes.get("Fractal_Depth")
    if not depth_attr:
        depth_attr = cloud_mesh.attributes.new(name="Fractal_Depth", type="FLOAT", domain="POINT")
    depth_attr.data.foreach_set("value", depths)

    # 写入自定义属性 Fractal_Iteration
    iter_attr = cloud_mesh.attributes.get("Fractal_Iteration")
    if not iter_attr:
        iter_attr = cloud_mesh.attributes.new(name="Fractal_Iteration", type="FLOAT", domain="POINT")
    iter_attr.data.foreach_set("value", iters)

    cloud_mesh.update()
    update_base_geometry(base_obj.data, f_type, mode, unit_scale)
    setup_fractal_instances_gn_robust(cloud_obj, base_obj)


# --- UI & Registration ---


class TaichiFractalProperties(bpy.types.PropertyGroup):
    fractal_type: bpy.props.EnumProperty(
        name="Type",
        items=[("SIERPINSKI", "Sierpinski", ""), ("MENGER", "Menger", "")],
        default="SIERPINSKI",
        update=update_mesh,
    )
    mode: bpy.props.EnumProperty(name="Mode", items=[("2D", "2D", ""), ("3D", "3D", "")], default="3D", update=update_mesh)
    level: bpy.props.IntProperty(name="Level", default=3, min=0, max=12, update=update_mesh)
    size: bpy.props.FloatProperty(name="Size", default=2.0, min=0.1, max=100.0, update=update_mesh)


class VIEW3D_PT_taichi_fractal_panel(bpy.types.Panel):
    bl_label = "Taichi Fractal GN Lab"
    bl_idname = "VIEW3D_PT_taichi_fractal"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout = self.layout
        props = context.scene.taichi_fractal_props
        layout.label(text="Auto-GN Fractal Engine", icon="NODETREE")
        layout.prop(props, "fractal_type", expand=True)
        layout.prop(props, "mode", expand=True)
        layout.prop(props, "level")
        layout.prop(props, "size")


def register():
    init_luts()
    bpy.utils.register_class(TaichiFractalProperties)
    bpy.utils.register_class(VIEW3D_PT_taichi_fractal_panel)
    bpy.types.Scene.taichi_fractal_props = bpy.props.PointerProperty(type=TaichiFractalProperties)


def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_taichi_fractal_panel)
    bpy.utils.unregister_class(TaichiFractalProperties)
    del bpy.types.Scene.taichi_fractal_props


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
