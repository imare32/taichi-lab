import bmesh
import bpy
import numpy as np
import taichi as ti

# =========================
# 0. Global Constants & Taichi Init
# =========================

ATTR_TENSION = "tension"
ATTR_EDGE_INDEX = "edge_index"
MAX_SOLVER_ITERS = 25

if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)

# =========================
# 1. Taichi Kernels (Parallel Computation)
# =========================

solve_stats = ti.field(ti.i32, shape=())


@ti.func
def sinh(x: float) -> float:
    return (ti.math.exp(x) - ti.math.exp(-x)) / 2.0


@ti.func
def cosh(x: float) -> float:
    return (ti.math.exp(x) + ti.math.exp(-x)) / 2.0


@ti.func
def newton_solve_w(r: float) -> float:
    """Solve sinh(w) / w = r via Newton-Raphson to find catenary parameter."""
    w = 0.0
    if r < 1.0001:
        w = 0.001
    elif r < 3.0:
        w = ti.math.sqrt(6.0 * (r - 1.0))
    else:
        w = ti.math.log(2.0 * r) + ti.math.log(ti.math.log(2.0 * r))

    iters = 0
    for i in range(MAX_SOLVER_ITERS):
        iters = i + 1
        f = sinh(w) - r * w
        df = cosh(w) - r
        if abs(df) < 1e-7:
            break
        delta = f / df
        delta = ti.math.clamp(delta, -2.0, 2.0)
        w_new = w - delta
        if abs(w_new - w) < 1e-6:
            w = w_new
            break
        w = w_new
        if w < 1e-5:
            w = 1e-5
    ti.atomic_max(solve_stats[None], iters)
    return w


@ti.kernel
def compute_catenaries_kernel(
    edges_v1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    edges_v2: ti.types.ndarray(dtype=ti.f32, ndim=2),
    cable_lengths: ti.types.ndarray(dtype=ti.f32, ndim=1),
    mode: int,
    strength: float,
    res: int,
    verts_out: ti.types.ndarray(dtype=ti.f32, ndim=2),
    tension_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    edge_idx_out: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    """Compute catenary curves for all edges in parallel."""
    n_edges = edges_v1.shape[0]
    PI_VAL = ti.math.pi

    for e in range(n_edges):
        p1 = ti.math.vec3(edges_v1[e, 0], edges_v1[e, 1], edges_v1[e, 2])
        p2 = ti.math.vec3(edges_v2[e, 0], edges_v2[e, 1], edges_v2[e, 2])
        L = cable_lengths[e]

        dist_3d = (p2 - p1).norm()
        dist_xy = ti.math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
        dz = p2.z - p1.z

        is_straight = L <= dist_3d + 1e-4
        is_degenerate = dist_xy < 1e-4
        a, h, k, s0, dir_x, dir_y = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        max_cosh = 1.0

        if not is_straight and not is_degenerate:
            s_total = ti.math.sqrt(L**2 - dz**2)
            r = s_total / dist_xy
            w = newton_solve_w(r)

            a = dist_xy / (2.0 * w)
            v_val = dz / s_total
            arsinh_v = ti.math.log(v_val + ti.math.sqrt(v_val**2 + 1.0))

            h = dist_xy / 2.0 - a * arsinh_v
            k = p1.z - a * cosh(h / a)

            s0 = a * sinh(-h / a)
            dir_x = (p2.x - p1.x) / dist_xy
            dir_y = (p2.y - p1.y) / dist_xy

            cosh_p1 = cosh(h / a)
            cosh_p2 = cosh((dist_xy - h) / a)
            max_cosh = ti.max(cosh_p1, cosh_p2)

        base = e * res
        for i in range(res):
            t = float(i) / (res - 1)
            idx = base + i

            if is_straight:
                pos = p1 + t * (p2 - p1)
                verts_out[idx, 0], verts_out[idx, 1], verts_out[idx, 2] = pos.x, pos.y, pos.z
                tension_out[idx] = 1.0
            elif is_degenerate:
                drop = (L - abs(dz)) / 2.0
                z = p1.z + t * dz
                t_val = 0.0
                if 0.0 < t < 1.0:
                    z -= drop * ti.math.sin(t * PI_VAL)
                    t_val = 1.0
                verts_out[idx, 0], verts_out[idx, 1], verts_out[idx, 2] = p1.x, p1.y, z
                tension_out[idx] = t_val
            else:
                target_s = 0.0
                if mode == 0:  # UNIFORM
                    target_s = t * L
                else:  # ADAPTIVE
                    s_u = t * L
                    theta1 = ti.math.atan2(s0, a)
                    theta2 = ti.math.atan2(s0 + L, a)
                    theta = theta1 + t * (theta2 - theta1)
                    s_a = a * ti.math.tan(theta) - s0
                    weight = ti.math.sin(t * PI_VAL)
                    local_strength = strength * weight
                    target_s = (1.0 - local_strength) * s_u + local_strength * s_a

                sinh_val = (target_s + s0) / a
                arsinh_val = ti.math.log(sinh_val + ti.math.sqrt(sinh_val**2 + 1.0))
                u = h + a * arsinh_val

                c_val = cosh((u - h) / a)
                z = a * c_val + k
                verts_out[idx, 0] = p1.x + u * dir_x
                verts_out[idx, 1] = p1.y + u * dir_y
                verts_out[idx, 2] = z

                t_norm = (c_val - 1.0) / (max_cosh - 1.0 + 1e-6)
                tension_out[idx] = ti.math.clamp(t_norm, 0.0, 1.0)

            edge_idx_out[idx] = e


# =========================
# 2. Geometry Utilities
# =========================


def extract_edges(obj_ref: bpy.types.Object, depsgraph=None):
    """Vectorized extraction of world-space edge endpoints, supporting modifiers/GN."""
    mat = np.array(obj_ref.matrix_world, dtype=np.float32)
    rot_scale = mat[:3, :3]
    loc = mat[:3, 3]

    if obj_ref.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj_ref.data)
        n_edges = len(bm.edges)
        if n_edges == 0:
            return None, None, None
        v1_local = np.empty((n_edges, 3), dtype=np.float32)
        v2_local = np.empty((n_edges, 3), dtype=np.float32)
        for i, edge in enumerate(bm.edges):
            v1_local[i] = edge.verts[0].co
            v2_local[i] = edge.verts[1].co
        edges_v1 = np.dot(v1_local, rot_scale.T) + loc
        edges_v2 = np.dot(v2_local, rot_scale.T) + loc
    else:
        # Use provided depsgraph to avoid recursive evaluation calls
        if depsgraph is None:
            depsgraph = bpy.context.evaluated_depsgraph_get()
        
        eval_obj = obj_ref.evaluated_get(depsgraph)
        temp_mesh = eval_obj.to_mesh()
        
        n_verts, n_edges = len(temp_mesh.vertices), len(temp_mesh.edges)
        if n_edges == 0:
            eval_obj.to_mesh_clear()
            return None, None, None
            
        coords = np.empty(n_verts * 3, dtype=np.float32)
        temp_mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(n_verts, 3)
        
        edge_indices = np.empty(n_edges * 2, dtype=np.int32)
        temp_mesh.edges.foreach_get("vertices", edge_indices)
        edge_indices = edge_indices.reshape(n_edges, 2)
        
        coords_w = np.dot(coords, rot_scale.T) + loc
        edges_v1 = coords_w[edge_indices[:, 0]]
        edges_v2 = coords_w[edge_indices[:, 1]]
        
        # Mandatory cleanup for temp mesh data
        eval_obj.to_mesh_clear()

    edge_lengths = np.linalg.norm(edges_v2 - edges_v1, axis=1).astype(np.float32)
    return edges_v1, edges_v2, edge_lengths


def setup_gn_viz(obj: bpy.types.Object, props):
    """Programmatically setup Geometry Nodes for curve sweeping."""
    gn_name = "Catenary_Viz"
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
        for i, (name, dtype) in enumerate([(ATTR_TENSION, "FLOAT"), (ATTR_EDGE_INDEX, "INT")]):
            att = nodes.new("GeometryNodeInputNamedAttribute")
            att.data_type = dtype
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


def build_mesh_topology(n_edges: int, res: int):
    """Pre-compute vertex count and edge connectivity for the generated mesh."""
    n_verts = n_edges * res
    n_edges_mesh = n_edges * (res - 1)
    edge_indices = np.empty(n_edges_mesh * 2, dtype=np.int32)
    for e in range(n_edges):
        base = e * res
        for i in range(res - 1):
            idx = (e * (res - 1) + i) * 2
            edge_indices[idx] = base + i
            edge_indices[idx + 1] = base + i + 1
    return n_verts, edge_indices


# =========================
# 3. Blender Mesh Sync (Object & Edit Mode)
# =========================


def sync_mesh_object(mesh, n_v, e_idx, verts, tension, e_id):
    """High-speed update using foreach_set."""
    if len(mesh.vertices) != n_v:
        mesh.clear_geometry()
        mesh.from_pydata([(0, 0, 0)] * n_v, [(e_idx[i], e_idx[i + 1]) for i in range(0, len(e_idx), 2)], [])

    mesh.vertices.foreach_set("co", verts.flatten())

    for attr_name, attr_type, data in [(ATTR_TENSION, "FLOAT", tension), (ATTR_EDGE_INDEX, "INT", e_id)]:
        if attr_name not in mesh.attributes:
            mesh.attributes.new(name=attr_name, type=attr_type, domain="POINT")
        mesh.attributes[attr_name].data.foreach_set("value", data)
    mesh.update()


def sync_mesh_edit(mesh, n_v, e_idx, verts, tension, e_id):
    """Surgical BMesh update for real-time Edit Mode feedback."""
    bm = bmesh.from_edit_mesh(mesh)
    if len(bm.verts) != n_v:
        bm.clear()
        for i in range(n_v):
            bm.verts.new()
        bm.verts.ensure_lookup_table()
        for i in range(0, len(e_idx), 2):
            bm.edges.new((bm.verts[e_idx[i]], bm.verts[e_idx[i + 1]]))
        bm.verts.layers.float.new(ATTR_TENSION)
        bm.verts.layers.int.new(ATTR_EDGE_INDEX)

    l_tension = bm.verts.layers.float.get(ATTR_TENSION) or bm.verts.layers.float.new(ATTR_TENSION)
    l_eidx = bm.verts.layers.int.get(ATTR_EDGE_INDEX) or bm.verts.layers.int.new(ATTR_EDGE_INDEX)

    for i, v in enumerate(bm.verts):
        v.co, v[l_tension], v[l_eidx] = verts[i], tension[i], int(e_id[i])
        v.select_set(True)

    bmesh.update_edit_mesh(mesh)


# =========================
# 4. State & Properties
# =========================


def on_dirty_update(self, context):
    self.is_dirty = True


def on_visual_update(self, context):
    """Sync visualization settings without triggering re-computation."""
    if self.obj_gen:
        setup_gn_viz(self.obj_gen, self)


class TAI_CatenaryProperties(bpy.types.PropertyGroup):
    obj_ref: bpy.props.PointerProperty(
        name="Reference", type=bpy.types.Object, poll=lambda s, o: o.type == "MESH", update=on_dirty_update
    )
    obj_gen: bpy.props.PointerProperty(name="Generated", type=bpy.types.Object, poll=lambda s, o: o.type == "MESH")
    length_mode: bpy.props.EnumProperty(
        name="Length Mode",
        items=[("SLACK", "Slack", "Multiplier"), ("ADD", "Add", "Fixed Offset")],
        default="SLACK",
        update=on_dirty_update,
    )
    slack: bpy.props.FloatProperty(name="Slack", default=3.0, min=1.0, soft_max=10.0, update=on_dirty_update)
    length_add: bpy.props.FloatProperty(name="Add", default=2.0, min=0.0, soft_max=10.0, update=on_dirty_update)
    resolution: bpy.props.IntProperty(name="Resolution", default=32, min=3, max=256, update=on_dirty_update)
    sampling_mode: bpy.props.EnumProperty(
        name="Sampling",
        items=[("UNIFORM", "Uniform", "Arc Length"), ("ADAPTIVE", "Adaptive", "Curvature Based")],
        default="ADAPTIVE",
        update=on_dirty_update,
    )
    sampling_strength: bpy.props.FloatProperty(name="Strength", default=0.75, min=0.0, max=1.0, update=on_dirty_update)
    bevel_radius: bpy.props.FloatProperty(
        name="Bevel Radius", default=0.02, min=0.0, soft_max=0.5, update=on_visual_update
    )
    bevel_resolution: bpy.props.IntProperty(
        name="Bevel Resolution", default=8, min=3, max=64, update=on_visual_update
    )
    is_dirty: bpy.props.BoolProperty(default=True)
    iter_count: bpy.props.IntProperty(name="Solver Iterations")
    edge_count: bpy.props.IntProperty(name="Source Edges")


# =========================
# 5. UI Panels & Operators
# =========================


class TAI_OT_init_catenary(bpy.types.Operator):
    bl_idname, bl_label = "taichi.init_catenary", "Initialize Catenary"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.taichi_catenary
        if not props.obj_ref:
            # Auto-create default ref polyline
            n, sp, h = 5, 1.0, 3.0
            v = [(i * sp - (n - 1) * sp / 2, 0.0, h) for i in range(n)]
            e = [(i, i + 1) for i in range(n - 1)]
            m = bpy.data.meshes.new("Catenary_Ref")
            m.from_pydata(v, e, [])
            o = bpy.data.objects.new("Catenary_Ref", m)
            context.collection.objects.link(o)
            props.obj_ref = o

        gen_name = f"{props.obj_ref.name}_catenary"
        if not props.obj_gen or props.obj_gen.name != gen_name:
            o_gen = bpy.data.objects.get(gen_name) or bpy.data.objects.new(gen_name, bpy.data.meshes.new(gen_name))
            if o_gen.name not in context.collection.objects:
                context.collection.objects.link(o_gen)
            props.obj_gen = o_gen

        update_catenary(context.scene)
        return {"FINISHED"}


class TAI_OT_select_catenary(bpy.types.Operator):
    bl_idname, bl_label = "taichi.select_catenary", "Select"
    target: bpy.props.EnumProperty(items=[("REF", "Reference", ""), ("GEN", "Generated", "")])

    def execute(self, context):
        props = context.scene.taichi_catenary
        obj = props.obj_ref if self.target == "REF" else props.obj_gen
        if obj:
            if context.object and context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            context.view_layer.objects.active = obj
        return {"FINISHED"}


class TAI_OT_toggle_ref_wire(bpy.types.Operator):
    bl_idname = "taichi.toggle_ref_wire"
    bl_label = "Toggle Reference Wire"
    bl_description = "Toggle the display type of the reference mesh between Wire and Textured"
    bl_options = {"UNDO"}

    def execute(self, context):
        props = context.scene.taichi_catenary
        obj = props.obj_ref
        if obj:
            obj.display_type = "WIRE" if obj.display_type != "WIRE" else "TEXTURED"
        return {"FINISHED"}


class TAI_PT_catenary_panel(bpy.types.Panel):
    bl_label = "Taichi Catenary"
    bl_idname = "TAI_PT_catenary_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout, props = self.layout, context.scene.taichi_catenary

        row = layout.row(align=True)
        row.prop(props, "obj_ref", text="")
        if props.obj_ref:
            is_wire = props.obj_ref.display_type == "WIRE"
            icon = "SHADING_WIRE" if is_wire else "SHADING_TEXTURE"
            row.operator(TAI_OT_toggle_ref_wire.bl_idname, text="", icon=icon)

        layout.operator(TAI_OT_init_catenary.bl_idname, icon="CURVE_PATH")

        if props.obj_gen:
            box = layout.box()
            box.label(text="Parameters", icon="SETTINGS")
            row = box.row(align=True)
            row.prop(props, "length_mode", expand=True)
            box.prop(props, "slack" if props.length_mode == "SLACK" else "length_add")

            box = layout.box()
            box.label(text="Sampling", icon="CURVE_PATH")
            box.prop(props, "resolution")
            row = box.row(align=True)
            row.prop(props, "sampling_mode", expand=True)
            if props.sampling_mode == "ADAPTIVE":
                box.prop(props, "sampling_strength", slider=True)

            box = layout.box()
            box.label(text="Visuals", icon="RESTRICT_RENDER_OFF")
            col = box.column(align=True)
            col.prop(props, "bevel_radius")
            col.prop(props, "bevel_resolution")

            box = layout.box()
            box.label(text="Components", icon="OBJECT_DATA")
            box.prop(props, "obj_ref", text="")
            box.prop(props, "obj_gen", text="")
            row = box.row(align=True)
            for t, l in [("REF", "Reference"), ("GEN", "Generated")]:
                row.operator(TAI_OT_select_catenary.bl_idname, text=l).target = t

            box = layout.box()
            col = box.column(align=True)
            row = col.row()
            row.label(text=f"Solver Steps: {props.iter_count}", icon="INFO")
            if props.iter_count >= MAX_SOLVER_ITERS:
                row.label(text="Limit Reached", icon="ERROR")
            col.label(text=f"Source Edges: {props.edge_count}", icon="LINENUMBERS_ON")

            # Data Interface Table
            sec = layout.box()
            sec.label(text="Data Interface", icon="SPREADSHEET")
            col = sec.column(align=True)
            for names in [
                ("Name", "Domain", "Type"),
                (ATTR_TENSION, "Point", "FLOAT"),
                (ATTR_EDGE_INDEX, "Point", "INT"),
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
# 6. Core Update Logic & Visualization
# =========================


def update_catenary(scene: bpy.types.Scene, depsgraph=None):
    """Coordinate the full catenary update cycle."""
    props = scene.taichi_catenary
    obj_ref = props.obj_ref
    obj_gen = props.obj_gen

    if not obj_ref or not obj_gen or not props.is_dirty:
        return

    # Sync visuals
    setup_gn_viz(obj_gen, props)

    # Extract edge data from reference mesh (passing depsgraph)
    e1, e2, lens = extract_edges(obj_ref, depsgraph=depsgraph)
    if e1 is None:
        return

    n_e, res = len(e1), props.resolution
    props.edge_count = n_e
    mode = 1 if props.sampling_mode == "ADAPTIVE" else 0
    c_lens = lens * props.slack if props.length_mode == "SLACK" else lens + props.length_add

    n_v, e_idx = build_mesh_topology(n_e, res)
    v_out, t_out, id_out = np.zeros((n_v, 3), np.float32), np.zeros(n_v, np.float32), np.zeros(n_v, np.int32)

    # Run Taichi kernel
    solve_stats[None] = 0
    compute_catenaries_kernel(e1, e2, c_lens, mode, props.sampling_strength, res, v_out, t_out, id_out)
    props.iter_count = solve_stats[None]

    # Sync to Blender mesh
    sync = sync_mesh_edit if obj_gen.mode == "EDIT" else sync_mesh_object
    sync(obj_gen.data, n_v, e_idx, v_out, t_out, id_out)
    
    props.is_dirty = False


# =========================
# 7. Lifecycle & Registration
# =========================


@bpy.app.handlers.persistent
def catenary_handler(scene, depsgraph):
    props = scene.taichi_catenary
    obj_ref = props.obj_ref
    if not obj_ref:
        return

    should_update = False
    # Standard update check for manual edits
    for update in depsgraph.updates:
        if update.id.name in [obj_ref.name, getattr(obj_ref.data, "name", "")]:
            should_update = True
            break

    if should_update:
        props.is_dirty = True

    # Pass the existing depsgraph to avoid re-triggering evaluation
    update_catenary(scene, depsgraph=depsgraph)


classes = (
    TAI_CatenaryProperties,
    TAI_OT_init_catenary,
    TAI_OT_select_catenary,
    TAI_OT_toggle_ref_wire,
    TAI_PT_catenary_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.taichi_catenary = bpy.props.PointerProperty(type=TAI_CatenaryProperties)
    bpy.app.handlers.depsgraph_update_post[:] = [
        h for h in bpy.app.handlers.depsgraph_update_post if h.__name__ != "catenary_handler"
    ]
    bpy.app.handlers.depsgraph_update_post.append(catenary_handler)


def unregister():
    bpy.app.handlers.depsgraph_update_post[:] = [
        h for h in bpy.app.handlers.depsgraph_update_post if h.__name__ != "catenary_handler"
    ]
    del bpy.types.Scene.taichi_catenary
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
