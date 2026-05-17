from math import gcd, pi

import bmesh
import bpy
import numpy as np
import taichi as ti

# =========================
# 0. Global Constants & Taichi Init
# =========================

ATTR_U = "u_map"
ATTR_V = "v_map"

if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)

# =========================
# 1. Taichi Kernels (Parallel Computation)
# =========================


@ti.kernel
def compute_torus_points_kernel(
    out_cos: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out_u: ti.types.ndarray(dtype=ti.f32, ndim=1),
    out_v: ti.types.ndarray(dtype=ti.f32, ndim=1),
    p: int,
    q: int,
    u_mult: int,
    v_mult: int,
    h: float,
    s: float,
    R_base: float,
    r_base: float,
    rPhase_base: float,
    sPhase: float,
    links: int,
    res: int,
):
    """Compute 3D coordinates and UV mapping for torus knot points."""
    R = R_base * s
    r = r_base * s
    da = 2 * pi / links / (res - 1)

    for l, n in ti.ndrange(links, res - 1):
        t = n * da
        linkPhase = 2 * pi / q * l
        rPhase = rPhase_base + linkPhase

        theta = p * t * u_mult + rPhase
        phi = q * t * v_mult + sPhase

        x = (R + r * ti.cos(phi)) * ti.cos(theta)
        y = (R + r * ti.cos(phi)) * ti.sin(theta)
        z = r * ti.sin(phi) * h

        idx = l * (res - 1) + n
        out_cos[idx, 0] = x
        out_cos[idx, 1] = y
        out_cos[idx, 2] = z
        
        u_val = theta / (2 * pi)
        v_val = phi / (2 * pi)
        out_u[idx] = ti.cast(u_val - ti.floor(u_val), ti.f32)
        out_v[idx] = ti.cast(v_val - ti.floor(v_val), ti.f32)


# =========================
# 2. Geometry Utilities
# =========================


def setup_gn_viz(obj: bpy.types.Object, props):
    """Programmatically setup Geometry Nodes for knot sweeping."""
    gn_name = "TorusKnot_Viz"
    nt = bpy.data.node_groups.get(gn_name)
    if not nt:
        nt = bpy.data.node_groups.new(name=gn_name, type="GeometryNodeTree")
        nt.is_modifier = True
        nt.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        nt.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        
        nodes, links = nt.nodes, nt.links
        inp, out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
        m2c = nodes.new("GeometryNodeMeshToCurve")
        sst = nodes.new("GeometryNodeCurveSplineType")
        sst.spline_type = "BEZIER"
        sh = nodes.new("GeometryNodeCurveSetHandles")
        sh.handle_type = "AUTO"
        sh.mute = True
        c2m = nodes.new("GeometryNodeCurveToMesh")
        circle = nodes.new("GeometryNodeCurvePrimitiveCircle")
        circle.name, circle.inputs[0].default_value = "Curve Circle", 8

        # Discoverability: Floating Named Attribute nodes
        for i, name in enumerate([ATTR_U, ATTR_V]):
            att = nodes.new("GeometryNodeInputNamedAttribute")
            att.data_type = "FLOAT"
            att.inputs[0].default_value = name
            att.location = (400, -200 - i * 150)

        links.new(inp.outputs[0], m2c.inputs[0])
        links.new(m2c.outputs[0], sst.inputs[0])
        links.new(sst.outputs[0], sh.inputs[0])
        links.new(sh.outputs[0], c2m.inputs[0])
        links.new(circle.outputs[0], c2m.inputs[1])
        links.new(c2m.outputs[0], out.inputs[0])

    mod = obj.modifiers.get(gn_name) or obj.modifiers.new(gn_name, "NODES")
    mod.show_in_editmode, mod.node_group = False, nt
    
    if props.bevel_radius == 0.0:
        mod.show_viewport = False
    else:
        mod.show_viewport = True
        if "Curve Circle" in nt.nodes:
            node_circle = nt.nodes["Curve Circle"]
            node_circle.inputs[4].default_value = props.bevel_radius
            node_circle.inputs[0].default_value = props.bevel_resolution


# =========================
# 3. Blender Mesh Sync (Object & Edit Mode)
# =========================


def sync_mesh_object(mesh, n_v, e_idx, verts, u_map, v_map):
    """High-speed update using foreach_set."""
    if len(mesh.vertices) != n_v:
        mesh.clear_geometry()
        mesh.from_pydata([(0, 0, 0)] * n_v, e_idx, [])

    mesh.vertices.foreach_set("co", verts.flatten())

    for attr_name, data in [(ATTR_U, u_map), (ATTR_V, v_map)]:
        if attr_name not in mesh.attributes:
            mesh.attributes.new(name=attr_name, type="FLOAT", domain="POINT")
        mesh.attributes[attr_name].data.foreach_set("value", data)
    mesh.update()


def sync_mesh_edit(mesh, n_v, e_idx, verts, u_map, v_map):
    """Surgical BMesh update for real-time Edit Mode feedback."""
    bm = bmesh.from_edit_mesh(mesh)
    if len(bm.verts) != n_v:
        bm.clear()
        for _ in range(n_v):
            bm.verts.new()
        bm.verts.ensure_lookup_table()
        for i in range(len(e_idx)):
            bm.edges.new((bm.verts[e_idx[i][0]], bm.verts[e_idx[i][1]]))
    
    l_u = bm.verts.layers.float.get(ATTR_U) or bm.verts.layers.float.new(ATTR_U)
    l_v = bm.verts.layers.float.get(ATTR_V) or bm.verts.layers.float.new(ATTR_V)

    for i, v in enumerate(bm.verts):
        v.co, v[l_u], v[l_v] = verts[i], u_map[i], v_map[i]

    bmesh.update_edit_mesh(mesh)


# =========================
# 4. State & Properties
# =========================


def on_dirty_update(self, context):
    """Trigger a full simulation update when a parametric property changes."""
    self.is_dirty = True
    update_torus_knot(context.scene)


def on_visual_update(self, context):
    """Sync visualization settings without triggering re-computation."""
    if self.obj_gen:
        setup_gn_viz(self.obj_gen, self)


class TAI_TorusKnotProperties(bpy.types.PropertyGroup):
    obj_gen: bpy.props.PointerProperty(name="Generated", type=bpy.types.Object, poll=lambda s, o: o.type == "MESH")

    torus_p: bpy.props.IntProperty(name="p", default=2, min=1, update=on_dirty_update)
    torus_q: bpy.props.IntProperty(name="q", default=3, min=1, update=on_dirty_update)
    flip_p: bpy.props.BoolProperty(name="Flip p", default=False, update=on_dirty_update)
    flip_q: bpy.props.BoolProperty(name="Flip q", default=False, update=on_dirty_update)
    multiple_links: bpy.props.BoolProperty(name="Multiple Links", default=True, update=on_dirty_update)

    torus_u: bpy.props.IntProperty(name="Rev. Multiplier", default=1, min=1, update=on_dirty_update)
    torus_v: bpy.props.IntProperty(name="Spin Multiplier", default=1, min=1, update=on_dirty_update)
    torus_rP: bpy.props.FloatProperty(name="Revolution Phase", default=0.0, update=on_dirty_update)
    torus_sP: bpy.props.FloatProperty(name="Spin Phase", default=0.0, update=on_dirty_update)

    torus_R: bpy.props.FloatProperty(name="Major Radius", default=1.0, min=0.0, update=on_dirty_update)
    torus_r: bpy.props.FloatProperty(name="Minor Radius", default=0.25, min=0.0, update=on_dirty_update)
    torus_s: bpy.props.FloatProperty(name="Scale", default=1.0, min=0.01, update=on_dirty_update)
    torus_h: bpy.props.FloatProperty(name="Height", default=1.0, update=on_dirty_update)

    torus_res: bpy.props.IntProperty(name="Resolution", default=100, min=3, update=on_dirty_update)
    adaptive_resolution: bpy.props.BoolProperty(name="Adaptive Resolution", default=False, update=on_dirty_update)

    bevel_radius: bpy.props.FloatProperty(
        name="Bevel Radius", default=0.04, min=0.0, subtype="DISTANCE", unit="LENGTH", update=on_visual_update
    )
    bevel_resolution: bpy.props.IntProperty(
        name="Bevel Resolution", default=8, min=3, max=64, update=on_visual_update
    )

    is_dirty: bpy.props.BoolProperty(default=True)
    vertex_count: bpy.props.IntProperty(name="Vertex Count")


# =========================
# 5. UI Panels & Operators
# =========================


class TAI_OT_init_torus_knot(bpy.types.Operator):
    bl_idname, bl_label = "taichi.init_torus_knot", "Initialize Torus Knot"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.taichi_torus_knot
        gen_name = "TorusKnot_Generated"

        if not props.obj_gen or props.obj_gen.name != gen_name:
            o_gen = bpy.data.objects.get(gen_name) or bpy.data.objects.new(gen_name, bpy.data.meshes.new(gen_name))
            if o_gen.name not in context.collection.objects:
                context.collection.objects.link(o_gen)
            props.obj_gen = o_gen

        props.is_dirty = True
        update_torus_knot(context.scene)
        return {"FINISHED"}


class TAI_OT_select_torus(bpy.types.Operator):
    bl_idname, bl_label = "taichi.select_torus", "Select"

    def execute(self, context):
        props = context.scene.taichi_torus_knot
        obj = props.obj_gen
        if obj:
            if context.object and context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            context.view_layer.objects.active = obj
        return {"FINISHED"}


class TAI_PT_torus_knot_panel(bpy.types.Panel):
    bl_label = "Taichi Torus Knot"
    bl_idname = "TAI_PT_torus_knot_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout, props = self.layout, context.scene.taichi_torus_knot

        layout.operator(TAI_OT_init_torus_knot.bl_idname, icon="CURVE_PATH")

        if props.obj_gen:
            box = layout.box()
            box.label(text="Knot Parameters", icon="STRANDS")
            col = box.column(align=True)
            row = col.row(align=True)
            row.prop(props, "torus_p")
            row.prop(props, "flip_p", text="", icon="ARROW_LEFTRIGHT")
            row = col.row(align=True)
            row.prop(props, "torus_q")
            row.prop(props, "flip_q", text="", icon="ARROW_LEFTRIGHT")
            col.prop(props, "multiple_links")

            box = layout.box()
            box.label(text="Math Modifiers", icon="MODIFIER")
            col = box.column(align=True)
            col.prop(props, "torus_u")
            col.prop(props, "torus_v")
            col.prop(props, "torus_rP")
            col.prop(props, "torus_sP")

            box = layout.box()
            box.label(text="Dimensions", icon="MESH_TORUS")
            col = box.column(align=True)
            col.prop(props, "torus_R")
            col.prop(props, "torus_r")
            col.prop(props, "torus_s")
            col.prop(props, "torus_h")

            box = layout.box()
            box.label(text="Visuals", icon="RESTRICT_RENDER_OFF")
            col = box.column(align=True)
            col.prop(props, "adaptive_resolution")
            row = col.row(align=True)
            row.prop(props, "torus_res", text="Manual Resolution")
            row.enabled = not props.adaptive_resolution
            col.prop(props, "bevel_radius")
            col.prop(props, "bevel_resolution")

            box = layout.box()
            box.label(text="Components", icon="OBJECT_DATA")
            box.prop(props, "obj_gen", text="")
            box.operator(TAI_OT_select_torus.bl_idname, text="Select Generated", icon="RESTRICT_SELECT_OFF")

            box = layout.box()
            box.label(text=f"Points: {props.vertex_count}", icon="INFO")

            # Data Interface Table
            sec = layout.box()
            sec.label(text="Data Interface", icon="SPREADSHEET")
            col = sec.column(align=True)
            for names in [
                ("Name", "Domain", "Type"),
                (ATTR_U, "Point", "FLOAT"),
                (ATTR_V, "Point", "FLOAT"),
            ]:
                row = col.row(align=True)
                box = row.box()
                box.scale_y = 0.5 if names[0] != "Name" else 0.75
                r = box.row(align=True)
                split = r.split(factor=0.5)
                split.label(text=names[0])
                split.label(text=names[1])
                split.label(text=names[2])


# =========================
# 6. Core Update Logic
# =========================


def update_torus_knot(scene: bpy.types.Scene):
    """Coordinate the full torus knot update cycle."""
    props = scene.taichi_torus_knot
    obj_gen = props.obj_gen

    if not obj_gen or not props.is_dirty:
        return

    # Sync visuals (GN setup)
    setup_gn_viz(obj_gen, props)

    p = props.torus_p * (-1 if props.flip_p else 1)
    q = props.torus_q * (-1 if props.flip_q else 1)
    links = gcd(props.torus_p, props.torus_q) if props.multiple_links else 1

    res = props.torus_res
    if props.adaptive_resolution:
        R, r = props.torus_R, props.torus_r
        p_abs, q_abs = abs(p), abs(q)
        maxTKLen = 2 * pi * np.sqrt(p_abs**2 * (R + r) ** 2 + q_abs**2 * r**2)
        minTKLen = 2 * pi * np.sqrt(p_abs**2 * (R - r) ** 2 + q_abs**2 * r**2)
        avgTKLen = (minTKLen + maxTKLen) / 2
        res = max(3, int(avgTKLen / links) * 8)

    n_v = links * (res - 1)
    props.vertex_count = n_v

    v_out = np.zeros((n_v, 3), dtype=np.float32)
    u_out = np.zeros(n_v, dtype=np.float32)
    v_out_map = np.zeros(n_v, dtype=np.float32)

    compute_torus_points_kernel(
        v_out, u_out, v_out_map,
        p, q, props.torus_u, props.torus_v,
        props.torus_h, props.torus_s,
        props.torus_R, props.torus_r,
        props.torus_rP, props.torus_sP,
        links, res
    )

    # Pre-compute edges
    e_idx = []
    for l in range(links):
        offset = l * (res - 1)
        for n in range(res - 1):
            e_idx.append((offset + n, offset + (n + 1) % (res - 1)))

    # Sync to Blender mesh
    sync = sync_mesh_edit if obj_gen.mode == "EDIT" else sync_mesh_object
    sync(obj_gen.data, n_v, e_idx, v_out, u_out, v_out_map)

    props.is_dirty = False


# =========================
# 7. Lifecycle & Registration
# =========================


classes = (
    TAI_TorusKnotProperties,
    TAI_OT_init_torus_knot,
    TAI_OT_select_torus,
    TAI_PT_torus_knot_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.taichi_torus_knot = bpy.props.PointerProperty(type=TAI_TorusKnotProperties)


def unregister():
    del bpy.types.Scene.taichi_torus_knot
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()

