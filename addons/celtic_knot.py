import bmesh
import bpy
import numpy as np
import taichi as ti

# =========================
# 0. Global Constants & Taichi Init
# =========================

ATTR_WEAVE = "weave_phase"
ATTR_LOOP = "loop_id"

if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)

# =========================
# 1. Taichi Kernels (Parallel Computation)
# =========================

@ti.kernel
def compute_knot_points(
    vertices: ti.types.ndarray(dtype=ti.f32, ndim=2),
    edge_verts: ti.types.ndarray(dtype=ti.i32, ndim=2),
    loop_edge_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
    loop_normals: ti.types.ndarray(dtype=ti.f32, ndim=2),
    prev_loop_normals: ti.types.ndarray(dtype=ti.f32, ndim=2),
    is_forward: ti.types.ndarray(dtype=ti.i32, ndim=1),
    weave_up: float,
    weave_down: float,
    out_cos: ti.types.ndarray(dtype=ti.f32, ndim=2),
):
    """Compute offset positions for the knot based on weaving direction."""
    for i in range(loop_edge_indices.shape[0]):
        edge_idx = loop_edge_indices[i]
        v1_idx = edge_verts[edge_idx, 0]
        v2_idx = edge_verts[edge_idx, 1]

        v1 = ti.Vector([vertices[v1_idx, 0], vertices[v1_idx, 1], vertices[v1_idx, 2]])
        v2 = ti.Vector([vertices[v2_idx, 0], vertices[v2_idx, 1], vertices[v2_idx, 2]])
        midpoint = (v1 + v2) * 0.5

        n1 = ti.Vector([loop_normals[i, 0], loop_normals[i, 1], loop_normals[i, 2]])
        n2 = ti.Vector([prev_loop_normals[i, 0], prev_loop_normals[i, 1], prev_loop_normals[i, 2]])
        
        # Safe normalization
        normal = n1 + n2
        norm_len = normal.norm()
        if norm_len > 1e-6:
            normal = normal / norm_len
        else:
            normal = ti.Vector([0.0, 0.0, 1.0])

        offset = weave_up if is_forward[i] == 1 else weave_down
        pos = midpoint + offset * normal

        out_cos[i, 0] = pos.x
        out_cos[i, 1] = pos.y
        out_cos[i, 2] = pos.z


# =========================
# 2. Geometry Utilities
# =========================

def get_mesh_data(obj_ref: bpy.types.Object, depsgraph=None):
    """Retrieves bmesh for traversal and numpy arrays for Taichi computation (World Space)."""
    mat = np.array(obj_ref.matrix_world, dtype=np.float32)
    rot_scale = mat[:3, :3]
    loc = mat[:3, 3]

    if obj_ref.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj_ref.data)
        bm.verts.ensure_lookup_table()
    else:
        if depsgraph is None:
            depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj_ref.evaluated_get(depsgraph)
        temp_mesh = eval_obj.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(temp_mesh)
        eval_obj.to_mesh_clear()
    
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Vectorized world-space coordinate conversion for Taichi
    local_coords = np.array([v.co for v in bm.verts], dtype=np.float32)
    world_coords = np.dot(local_coords, rot_scale.T) + loc
    
    edge_verts = np.array([[e.verts[0].index, e.verts[1].index] for e in bm.edges], dtype=np.int32)

    return bm, world_coords, edge_verts


def traverse_knot(bm):
    """Traverses the mesh faces to find continuous loops for the celtic knot."""
    loops_entered = {}
    loops_exited = {}

    def ignorable_loop(loop):
        return len(loop.link_loops) == 0

    splines = []

    for face in bm.faces:
        for start_loop in face.loops:
            if ignorable_loop(start_loop):
                continue

            # Forward traversal
            if start_loop not in loops_exited:
                path = []
                loop = start_loop
                forward = True
                while True:
                    if forward:
                        if loop in loops_exited:
                            break
                        loops_exited[loop] = True
                        while True:
                            loop = loop.link_loop_next
                            if not ignorable_loop(loop):
                                break
                        loops_entered[loop] = True
                        prev_loop = loop
                        v = loop.vert.index
                        loop = loop.link_loops[0]
                        forward = loop.vert.index == v
                        path.append((loop, prev_loop, True))
                    else:
                        if loop in loops_entered:
                            break
                        loops_entered[loop] = True
                        while True:
                            v = loop.vert.index
                            loop = loop.link_loop_prev
                            if not ignorable_loop(loop):
                                break
                        loops_exited[loop] = True
                        prev_loop = loop
                        loop = loop.link_loops[-1]
                        forward = loop.vert.index == v
                        path.append((loop, prev_loop, False))
                if path:
                    splines.append(path)

            # Backward traversal
            if start_loop not in loops_entered:
                path = []
                loop = start_loop
                forward = False
                while True:
                    if forward:
                        if loop in loops_exited:
                            break
                        loops_exited[loop] = True
                        while True:
                            loop = loop.link_loop_next
                            if not ignorable_loop(loop):
                                break
                        loops_entered[loop] = True
                        prev_loop = loop
                        v = loop.vert.index
                        loop = loop.link_loops[0]
                        forward = loop.vert.index == v
                        path.append((loop, prev_loop, True))
                    else:
                        if loop in loops_entered:
                            break
                        loops_entered[loop] = True
                        while True:
                            v = loop.vert.index
                            loop = loop.link_loop_prev
                            if not ignorable_loop(loop):
                                break
                        loops_exited[loop] = True
                        prev_loop = loop
                        loop = loop.link_loops[-1]
                        forward = loop.vert.index == v
                        path.append((loop, prev_loop, False))
                if path:
                    splines.append(path)

    return splines


def setup_gn_viz(obj: bpy.types.Object, props):
    """Programmatically setup Geometry Nodes for knot sweeping."""
    gn_name = "CelticKnot_Viz"
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
        c2m = nodes.new("GeometryNodeCurveToMesh")
        circle = nodes.new("GeometryNodeCurvePrimitiveCircle")
        circle.name, circle.inputs[0].default_value = "Curve Circle", 8

        # Discoverability: Floating Named Attribute nodes
        for i, (name, dtype) in enumerate([(ATTR_WEAVE, "FLOAT"), (ATTR_LOOP, "INT")]):
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


# =========================
# 3. Blender Mesh Sync (Object & Edit Mode)
# =========================

def sync_mesh_object(mesh, n_v, e_idx, verts, weave_phase, loop_id):
    """High-speed update using foreach_set."""
    if len(mesh.vertices) != n_v:
        mesh.clear_geometry()
        mesh.from_pydata([(0, 0, 0)] * n_v, e_idx, [])

    mesh.vertices.foreach_set("co", verts.flatten())

    for attr_name, attr_type, data in [(ATTR_WEAVE, "FLOAT", weave_phase), (ATTR_LOOP, "INT", loop_id)]:
        if attr_name not in mesh.attributes:
            mesh.attributes.new(name=attr_name, type=attr_type, domain="POINT")
        mesh.attributes[attr_name].data.foreach_set("value", data)
    mesh.update()


def sync_mesh_edit(mesh, n_v, e_idx, verts, weave_phase, loop_id):
    """Surgical BMesh update for real-time Edit Mode feedback."""
    bm = bmesh.from_edit_mesh(mesh)
    if len(bm.verts) != n_v:
        bm.clear()
        for _ in range(n_v):
            bm.verts.new()
        bm.verts.ensure_lookup_table()
        for i in range(len(e_idx)):
            bm.edges.new((bm.verts[e_idx[i][0]], bm.verts[e_idx[i][1]]))
    
    l_weave = bm.verts.layers.float.get(ATTR_WEAVE) or bm.verts.layers.float.new(ATTR_WEAVE)
    l_loop = bm.verts.layers.int.get(ATTR_LOOP) or bm.verts.layers.int.new(ATTR_LOOP)

    for i, v in enumerate(bm.verts):
        v.co, v[l_weave], v[l_loop] = verts[i], weave_phase[i], int(loop_id[i])

    bmesh.update_edit_mesh(mesh)


# =========================
# 4. State & Properties
# =========================

def on_dirty_update(self, context):
    self.is_dirty = True

def on_visual_update(self, context):
    if self.obj_gen:
        setup_gn_viz(self.obj_gen, self)

class TAI_CelticKnotProperties(bpy.types.PropertyGroup):
    obj_ref: bpy.props.PointerProperty(
        name="Reference", type=bpy.types.Object, poll=lambda s, o: o.type == "MESH", update=on_dirty_update
    )
    obj_gen: bpy.props.PointerProperty(name="Generated", type=bpy.types.Object, poll=lambda s, o: o.type == "MESH")
    
    weave_up: bpy.props.FloatProperty(
        name="Weave Up", default=0.05, subtype="DISTANCE", unit="LENGTH", update=on_dirty_update
    )
    weave_down: bpy.props.FloatProperty(
        name="Weave Down", default=-0.05, subtype="DISTANCE", unit="LENGTH", update=on_dirty_update
    )
    bevel_radius: bpy.props.FloatProperty(
        name="Bevel Radius", default=0.05, min=0.0, subtype="DISTANCE", unit="LENGTH", update=on_visual_update
    )
    bevel_resolution: bpy.props.IntProperty(
        name="Bevel Resolution", default=8, min=3, max=64, update=on_visual_update
    )
    loop_count: bpy.props.IntProperty(name="Loop Count")
    is_dirty: bpy.props.BoolProperty(default=True)


# =========================
# 5. UI Panels & Operators
# =========================

class TAI_OT_init_celtic_knot(bpy.types.Operator):
    bl_idname, bl_label = "taichi.init_celtic_knot", "Initialize Celtic Knot"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.taichi_celtic_knot
        if not props.obj_ref:
            # Create a default reference subdivided plane
            bm = bmesh.new()
            bmesh.ops.create_grid(bm, x_segments=4, y_segments=4, size=1.0)
            m = bpy.data.meshes.new("Celtic_Ref")
            bm.to_mesh(m)
            bm.free()
            o = bpy.data.objects.new("Celtic_Ref", m)
            context.collection.objects.link(o)
            props.obj_ref = o

        gen_name = f"{props.obj_ref.name}_knot"
        if not props.obj_gen or props.obj_gen.name != gen_name:
            o_gen = bpy.data.objects.get(gen_name) or bpy.data.objects.new(gen_name, bpy.data.meshes.new(gen_name))
            if o_gen.name not in context.collection.objects:
                context.collection.objects.link(o_gen)
            props.obj_gen = o_gen

        update_knot(context.scene)
        return {"FINISHED"}


class TAI_OT_select_celtic(bpy.types.Operator):
    bl_idname, bl_label = "taichi.select_celtic", "Select"
    target: bpy.props.EnumProperty(items=[("REF", "Reference", ""), ("GEN", "Generated", "")])

    def execute(self, context):
        props = context.scene.taichi_celtic_knot
        obj = props.obj_ref if self.target == "REF" else props.obj_gen
        if obj:
            if context.object and context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            context.view_layer.objects.active = obj
        return {"FINISHED"}


class TAI_OT_toggle_ref_wire(bpy.types.Operator):
    bl_idname, bl_label = "taichi.toggle_ref_wire", "Toggle Reference Wire"
    bl_options = {"UNDO"}

    def execute(self, context):
        props = context.scene.taichi_celtic_knot
        if props.obj_ref:
            props.obj_ref.display_type = "WIRE" if props.obj_ref.display_type != "WIRE" else "TEXTURED"
        return {"FINISHED"}


class TAI_PT_celtic_panel(bpy.types.Panel):
    bl_label = "Taichi Celtic Knot"
    bl_idname = "TAI_PT_celtic_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout, props = self.layout, context.scene.taichi_celtic_knot

        row = layout.row(align=True)
        row.prop(props, "obj_ref", text="")
        if props.obj_ref:
            is_wire = props.obj_ref.display_type == "WIRE"
            row.operator(TAI_OT_toggle_ref_wire.bl_idname, text="", icon="SHADING_WIRE" if is_wire else "SHADING_TEXTURE")

        layout.operator(TAI_OT_init_celtic_knot.bl_idname, icon="CURVE_PATH")

        if props.obj_gen:
            box = layout.box()
            box.label(text="Parameters", icon="SETTINGS")
            col = box.column(align=True)
            col.prop(props, "weave_up")
            col.prop(props, "weave_down")

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
                row.operator(TAI_OT_select_celtic.bl_idname, text=l).target = t

            box = layout.box()
            row = box.row()
            row.label(text=f"Loops Found: {props.loop_count}", icon="INFO")

            # Data Interface Table
            sec = layout.box()
            sec.label(text="Data Interface", icon="SPREADSHEET")
            col = sec.column(align=True)
            for names in [
                ("Name", "Domain", "Type"),
                (ATTR_WEAVE, "Point", "FLOAT"),
                (ATTR_LOOP, "Point", "INT"),
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

def update_knot(scene: bpy.types.Scene, depsgraph=None):
    """Coordinate the full celtic knot update cycle."""
    props = scene.taichi_celtic_knot
    obj_ref = props.obj_ref
    obj_gen = props.obj_gen

    if not obj_ref or not obj_gen or not props.is_dirty:
        return

    # Sync visuals (GN setup)
    setup_gn_viz(obj_gen, props)

    # Extract mesh data
    bm_source, verts_source, edge_verts_source = get_mesh_data(obj_ref, depsgraph=depsgraph)
    splines_data = traverse_knot(bm_source)
    props.loop_count = len(splines_data)

    all_cos = []
    all_edges = []
    all_phases = []
    all_loop_ids = []
    v_offset = 0

    num_loops = len(splines_data)

    for idx, path in enumerate(splines_data):
        num_pts = len(path)
        loop_edge_indices = np.array([p[0].edge.index for p in path], dtype=np.int32)
        loop_normals = np.array([p[0].calc_normal() for p in path], dtype=np.float32)
        prev_loop_normals = np.array([p[1].calc_normal() for p in path], dtype=np.float32)
        is_forward = np.array([1 if p[2] else 0 for p in path], dtype=np.int32)

        out_cos = np.zeros((num_pts, 3), dtype=np.float32)

        compute_knot_points(
            verts_source,
            edge_verts_source,
            loop_edge_indices,
            loop_normals,
            prev_loop_normals,
            is_forward,
            props.weave_up,
            props.weave_down,
            out_cos,
        )

        all_cos.append(out_cos)
        for i in range(num_pts):
            all_edges.append((v_offset + i, v_offset + (i + 1) % num_pts))
        
        all_phases.append(1.0 - is_forward.astype(np.float32))
        all_loop_ids.append(np.full(num_pts, idx, dtype=np.int32))

        v_offset += num_pts

    if not all_cos:
        props.is_dirty = False
        return

    v_out = np.vstack(all_cos)
    e_idx = np.array(all_edges, dtype=np.int32)
    weave_out = np.concatenate(all_phases)
    loop_out = np.concatenate(all_loop_ids)
    n_v = v_out.shape[0]

    # Sync to Blender mesh
    sync = sync_mesh_edit if obj_gen.mode == "EDIT" else sync_mesh_object
    sync(obj_gen.data, n_v, e_idx, v_out, weave_out, loop_out)
    
    if obj_ref.mode != "EDIT":
        bm_source.free()
    
    props.is_dirty = False


# =========================
# 7. Lifecycle & Registration
# =========================

@bpy.app.handlers.persistent
def celtic_handler(scene, depsgraph):
    props = scene.taichi_celtic_knot
    obj_ref = props.obj_ref
    if not obj_ref:
        return

    should_update = False
    for update in depsgraph.updates:
        if update.id.name in [obj_ref.name, getattr(obj_ref.data, "name", "")]:
            should_update = True
            break

    if should_update:
        props.is_dirty = True

    update_knot(scene, depsgraph=depsgraph)


classes = (
    TAI_CelticKnotProperties,
    TAI_OT_init_celtic_knot,
    TAI_OT_select_celtic,
    TAI_OT_toggle_ref_wire,
    TAI_PT_celtic_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.taichi_celtic_knot = bpy.props.PointerProperty(type=TAI_CelticKnotProperties)
    
    bpy.app.handlers.depsgraph_update_post[:] = [
        h for h in bpy.app.handlers.depsgraph_update_post if h.__name__ != "celtic_handler"
    ]
    bpy.app.handlers.depsgraph_update_post.append(celtic_handler)

def unregister():
    bpy.app.handlers.depsgraph_update_post[:] = [
        h for h in bpy.app.handlers.depsgraph_update_post if h.__name__ != "celtic_handler"
    ]
    del bpy.types.Scene.taichi_celtic_knot
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__" or __name__ == "<run_path>":
    register()
