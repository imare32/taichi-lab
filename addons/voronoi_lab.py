import bmesh
import bpy
import mathutils
import numpy as np
import taichi as ti
from math import atan2

# =========================
# 0. Global Constants & Taichi Init
# =========================

ATTR_CELL_INDEX = "cell_index"

if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)

# =========================
# 1. Taichi Kernels (Parallel Sampling)
# =========================


@ti.kernel
def sample_bezier_segments_kernel(
    p0: ti.types.ndarray(dtype=ti.f32, ndim=2),
    p1: ti.types.ndarray(dtype=ti.f32, ndim=2),
    p2: ti.types.ndarray(dtype=ti.f32, ndim=2),
    p3: ti.types.ndarray(dtype=ti.f32, ndim=2),
    res: int,
    verts_out: ti.types.ndarray(dtype=ti.f32, ndim=2),
    cell_idx_in: ti.types.ndarray(dtype=ti.i32, ndim=1),
    cell_idx_out: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    """Parallel sampling of Cubic Bezier segments."""
    n_segments = p0.shape[0]
    for s in range(n_segments):
        v0 = ti.Vector([p0[s, 0], p0[s, 1], p0[s, 2]])
        v1 = ti.Vector([p1[s, 0], p1[s, 1], p1[s, 2]])
        v2 = ti.Vector([p2[s, 0], p2[s, 1], p2[s, 2]])
        v3 = ti.Vector([p3[s, 0], p3[s, 1], p3[s, 2]])
        c_idx = cell_idx_in[s]

        base = s * res
        for i in range(res):
            t = float(i) / (res - 1)
            # Cubic Bezier Equation
            pos = (1 - t) ** 3 * v0 + 3 * (1 - t) ** 2 * t * v1 + 3 * (1 - t) * t**2 * v2 + t**3 * v3

            idx = base + i
            verts_out[idx, 0], verts_out[idx, 1], verts_out[idx, 2] = pos.x, pos.y, pos.z
            cell_idx_out[idx] = c_idx


# =========================
# 2. Geometry Utilities (Python/Mathutils)
# =========================


def get_circumcenter(a, b, c):
    """Calculate circumcenter of a 2D triangle."""
    d = 2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))
    if abs(d) < 1e-9:
        return (a + b + c) / 3.0
    ux = ((a.x**2 + a.y**2) * (b.y - c.y) + (b.x**2 + b.y**2) * (c.y - a.y) + (c.x**2 + c.y**2) * (a.y - b.y)) / d
    uy = ((a.x**2 + a.y**2) * (c.x - b.x) + (b.x**2 + b.y**2) * (a.x - c.x) + (c.x**2 + c.y**2) * (b.x - a.x)) / d
    return mathutils.Vector((ux, uy))


def extract_seeds(obj_ref: bpy.types.Object, depsgraph=None):
    """Vectorized extraction of world-space vertex coordinates, supporting modifiers/GN."""
    mat = np.array(obj_ref.matrix_world, dtype=np.float32)
    rot_scale = mat[:3, :3]
    loc = mat[:3, 3]

    if obj_ref.mode == "EDIT":
        bm = bmesh.from_edit_mesh(obj_ref.data)
        n_verts = len(bm.verts)
        if n_verts == 0:
            return []
        v_local = np.array([v.co for v in bm.verts], dtype=np.float32)
        v_world = np.dot(v_local, rot_scale.T) + loc
    else:
        if depsgraph is None:
            depsgraph = bpy.context.evaluated_depsgraph_get()

        eval_obj = obj_ref.evaluated_get(depsgraph)
        temp_mesh = eval_obj.to_mesh()

        n_verts = len(temp_mesh.vertices)
        if n_verts == 0:
            eval_obj.to_mesh_clear()
            return []

        coords = np.empty(n_verts * 3, dtype=np.float32)
        temp_mesh.vertices.foreach_get("co", coords)
        coords = coords.reshape(n_verts, 3)

        v_world = np.dot(coords, rot_scale.T) + loc
        eval_obj.to_mesh_clear()

    return [mathutils.Vector(v) for v in v_world]


def extract_voronoi_cells(seeds_world):
    """Build Voronoi polygons using Delaunay CDT and circumcenters."""
    if not seeds_world:
        return []
    seeds_2d = [s.xy for s in seeds_world]

    # 1. Padded Bounding Box
    min_v = mathutils.Vector((min(p.x for p in seeds_2d), min(p.y for p in seeds_2d)))
    max_v = mathutils.Vector((max(p.x for p in seeds_2d), max(p.y for p in seeds_2d)))
    diag = (max_v - min_v).length or 1.0
    margin = diag * 10.0

    bbox_points = [
        mathutils.Vector((min_v.x - margin, min_v.y - margin)),
        mathutils.Vector((max_v.x + margin, min_v.y - margin)),
        mathutils.Vector((max_v.x + margin, max_v.y + margin)),
        mathutils.Vector((min_v.x - margin, max_v.y + margin)),
    ]

    n_orig = len(seeds_2d)
    all_points = seeds_2d + bbox_points
    res = mathutils.geometry.delaunay_2d_cdt(all_points, [], [], 0, 1e-5)
    out_verts, out_edges, out_faces, orig_v, orig_e, orig_f = res

    seed_to_out_idx = {}
    for j, source_indices in enumerate(orig_v):
        for si in source_indices:
            if si < n_orig:
                seed_to_out_idx[si] = j

    vert_to_faces = [[] for _ in range(len(out_verts))]
    for f_idx, face in enumerate(out_faces):
        for v_idx in face:
            vert_to_faces[v_idx].append(f_idx)

    face_centers = []
    for face in out_faces:
        face_centers.append(get_circumcenter(out_verts[face[0]], out_verts[face[1]], out_verts[face[2]]))

    cells = []
    for i in range(n_orig):
        out_idx = seed_to_out_idx.get(i)
        if out_idx is None:
            continue
        incident_faces = vert_to_faces[out_idx]
        if not incident_faces:
            continue

        seed_co_2d = out_verts[out_idx]
        poly_verts = [face_centers[f] for f in incident_faces]
        # Sort CCW around seed
        poly_verts.sort(key=lambda p: atan2(p.y - seed_co_2d.y, p.x - seed_co_2d.x))

        cells.append({"seed": seeds_world[i], "verts": poly_verts})
    return cells


def retract_polygon(poly_verts, seed_co_2d, dist):
    """Offset polygon edges inward (2D logic)."""
    if dist <= 0:
        return poly_verts
    n = len(poly_verts)
    new_edges = []
    for i in range(n):
        v1, v2 = poly_verts[i], poly_verts[(i + 1) % n]
        edge_dir = (v2 - v1).normalized()
        perp = mathutils.Vector((-edge_dir.y, edge_dir.x))
        center = (v1 + v2) * 0.5
        if (seed_co_2d - center).dot(perp) < 0:
            perp = -perp
        offset = perp * dist
        new_edges.append((v1 + offset, v2 + offset))

    final_verts = []
    for i in range(n):
        e1, e2 = new_edges[i], new_edges[(i + 1) % n]
        inter = mathutils.geometry.intersect_line_line_2d(e1[0], e1[1], e2[0], e2[1])
        final_verts.append(mathutils.Vector(inter) if inter else e1[1])
    return final_verts


def clip_poly_to_rect(poly_verts, clip_min, clip_max):
    """Clip a convex polygon to an axis-aligned rectangle (Sutherland-Hodgman)."""

    def clip(subject_poly, coord, value, is_upper):
        result = []
        if not subject_poly:
            return result
        for i in range(len(subject_poly)):
            p1 = subject_poly[i]
            p2 = subject_poly[(i + 1) % len(subject_poly)]

            v1, v2 = p1[coord], p2[coord]
            in1 = v1 <= value if is_upper else v1 >= value
            in2 = v2 <= value if is_upper else v2 >= value

            if in1:
                if in2:
                    result.append(p2)
                else:
                    t = (value - v1) / (v2 - v1)
                    result.append(p1 + t * (p2 - p1))
            else:
                if in2:
                    t = (value - v1) / (v2 - v1)
                    result.append(p1 + t * (p2 - p1))
                    result.append(p2)
        return result

    res = poly_verts
    res = clip(res, 0, clip_min.x, False)
    res = clip(res, 0, clip_max.x, True)
    res = clip(res, 1, clip_min.y, False)
    res = clip(res, 1, clip_max.y, True)
    return res


def setup_gn_viz(obj: bpy.types.Object, props):
    """Programmatically setup Geometry Nodes for Voronoi sweeping."""
    gn_name = "VoronoiLab_Viz"
    nt = bpy.data.node_groups.get(gn_name)
    if not nt:
        nt = bpy.data.node_groups.new(name=gn_name, type="GeometryNodeTree")
        nt.is_modifier = True
        nt.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        nt.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        nodes, links = nt.nodes, nt.links
        inp, out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
        m2c = nodes.new("GeometryNodeMeshToCurve")
        c2m = nodes.new("GeometryNodeCurveToMesh")
        circle = nodes.new("GeometryNodeCurvePrimitiveCircle")
        circle.name, circle.inputs[0].default_value = "Curve Circle", 8

        att = nodes.new("GeometryNodeInputNamedAttribute")
        att.data_type = "INT"
        att.inputs[0].default_value = ATTR_CELL_INDEX
        att.location = (400, -200)

        links.new(inp.outputs[0], m2c.inputs[0])
        links.new(m2c.outputs[0], c2m.inputs[0])
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
# 3. Blender Mesh Sync
# =========================


def sync_mesh_object(mesh, n_v, e_idx, verts, c_id):
    if len(mesh.vertices) != n_v:
        mesh.clear_geometry()
        mesh.from_pydata([(0, 0, 0)] * n_v, [(e_idx[i], e_idx[i + 1]) for i in range(0, len(e_idx), 2)], [])
    mesh.vertices.foreach_set("co", verts.flatten())
    if ATTR_CELL_INDEX not in mesh.attributes:
        mesh.attributes.new(name=ATTR_CELL_INDEX, type="INT", domain="POINT")
    mesh.attributes[ATTR_CELL_INDEX].data.foreach_set("value", c_id)
    mesh.update()


def sync_mesh_edit(mesh, n_v, e_idx, verts, c_id):
    bm = bmesh.from_edit_mesh(mesh)
    if len(bm.verts) != n_v:
        bm.clear()
        for _ in range(n_v):
            bm.verts.new()
        bm.verts.ensure_lookup_table()
        for i in range(0, len(e_idx), 2):
            bm.edges.new((bm.verts[e_idx[i]], bm.verts[e_idx[i + 1]]))
    l_cid = bm.verts.layers.int.get(ATTR_CELL_INDEX) or bm.verts.layers.int.new(ATTR_CELL_INDEX)
    for i, v in enumerate(bm.verts):
        v.co, v[l_cid] = verts[i], int(c_id[i])
    bmesh.update_edit_mesh(mesh)


# =========================
# 4. State & Properties
# =========================


def on_dirty_update(self, context):
    self.is_dirty = True
    # Force immediate update when UI properties change
    if context and hasattr(context, "scene"):
        update_voronoi(context.scene)
        # Tag all 3D viewports for redrawing
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class TAI_VoronoiProperties(bpy.types.PropertyGroup):
    obj_ref: bpy.props.PointerProperty(
        name="Reference", type=bpy.types.Object, poll=lambda s, o: o.type == "MESH", update=on_dirty_update
    )
    obj_gen: bpy.props.PointerProperty(name="Generated", type=bpy.types.Object, poll=lambda s, o: o.type == "MESH")
    cells_shape: bpy.props.EnumProperty(
        name="Shape",
        items=[("EDGES", "Edges", ""), ("QUADRATIC", "Quadratic", ""), ("CUBIC", "Cubic", "")],
        default="CUBIC",
        update=on_dirty_update,
    )
    cells_space: bpy.props.FloatProperty(name="Spacing", default=0.0, min=0.0, soft_max=1.0, update=on_dirty_update)
    min_edge: bpy.props.FloatProperty(name="Min Edge", default=0.0, min=0.0, soft_max=0.5, update=on_dirty_update)
    resolution: bpy.props.IntProperty(name="Resolution", default=10, min=2, max=64, update=on_dirty_update)
    boundary_type: bpy.props.EnumProperty(
        name="Boundary",
        items=[
            ("NONE", "None", "No clipping"),
            ("AUTO", "Auto", "Clip to seed bounds"),
            ("FIXED", "Fixed", "Clip to a fixed area"),
        ],
        default="AUTO",
        update=on_dirty_update,
    )
    clip_margin: bpy.props.FloatProperty(
        name="Clip Margin",
        description="Margin of the clipping boundary relative to seed bounds",
        default=0.1,
        min=0.0,
        soft_max=2.0,
        update=on_dirty_update,
    )
    boundary_width: bpy.props.FloatProperty(
        name="Width", description="Width of the fixed boundary", default=2.0, min=0.01, update=on_dirty_update
    )
    boundary_height: bpy.props.FloatProperty(
        name="Height", description="Height of the fixed boundary", default=2.0, min=0.01, update=on_dirty_update
    )
    use_max_radius: bpy.props.BoolProperty(
        name="Use Max Radius",
        description="Limit cell size to a maximum radius (Soap Bubble effect)",
        default=True,
        update=on_dirty_update,
    )
    max_radius: bpy.props.FloatProperty(
        name="Max Radius", description="Maximum radius of a free cell", default=0.5, min=0.01, update=on_dirty_update
    )
    bevel_radius: bpy.props.FloatProperty(
        name="Bevel Radius", default=0.0, min=0.0, update=lambda s, c: setup_gn_viz(s.obj_gen, s) if s.obj_gen else None
    )
    bevel_resolution: bpy.props.IntProperty(
        name="Bevel Res",
        default=8,
        min=3,
        max=32,
        update=lambda s, c: setup_gn_viz(s.obj_gen, s) if s.obj_gen else None,
    )
    is_dirty: bpy.props.BoolProperty(default=True)


# =========================
# 5. UI Panels & Operators
# =========================


class TAI_OT_init_voronoi(bpy.types.Operator):
    bl_idname, bl_label = "taichi.init_voronoi", "Initialize Voronoi"

    def execute(self, context):
        props = context.scene.taichi_voronoi
        if not props.obj_ref:
            m = bpy.data.meshes.new("Voronoi_Ref")
            bm = bmesh.new()
            bmesh.ops.create_grid(bm, x_segments=3, y_segments=3, size=1.0)
            bm.to_mesh(m)
            bm.free()
            o = bpy.data.objects.new("Voronoi_Ref", m)
            context.collection.objects.link(o)
            props.obj_ref = o
        gen_name = f"{props.obj_ref.name}_voro"
        if not props.obj_gen:
            o_gen = bpy.data.objects.get(gen_name) or bpy.data.objects.new(gen_name, bpy.data.meshes.new(gen_name))
            if o_gen.name not in context.collection.objects:
                context.collection.objects.link(o_gen)
            props.obj_gen = o_gen
        props.is_dirty = True
        return {"FINISHED"}


class TAI_OT_toggle_ref_wire(bpy.types.Operator):
    bl_idname, bl_label = "taichi.toggle_ref_wire", "Toggle Reference Wire"

    def execute(self, context):
        obj = context.scene.taichi_voronoi.obj_ref
        if obj:
            obj.display_type = "WIRE" if obj.display_type != "WIRE" else "TEXTURED"
        return {"FINISHED"}


class TAI_OT_select_voronoi(bpy.types.Operator):
    bl_idname, bl_label = "taichi.select_voronoi", "Select"
    target: bpy.props.EnumProperty(items=[("REF", "Reference", ""), ("GEN", "Generated", "")])

    def execute(self, context):
        props = context.scene.taichi_voronoi
        obj = props.obj_ref if self.target == "REF" else props.obj_gen
        if obj:
            if context.object and context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            context.view_layer.objects.active = obj
        return {"FINISHED"}


class TAI_PT_voronoi_panel(bpy.types.Panel):
    bl_label, bl_idname, bl_space_type, bl_region_type, bl_category = (
        "Taichi Voronoi",
        "TAI_PT_voronoi",
        "VIEW_3D",
        "UI",
        "Taichi",
    )

    def draw(self, context):
        layout, props = self.layout, context.scene.taichi_voronoi
        row = layout.row(align=True)
        row.prop(props, "obj_ref", text="")
        if props.obj_ref:
            icon = "SHADING_WIRE" if props.obj_ref.display_type == "WIRE" else "SHADING_TEXTURE"
            row.operator("taichi.toggle_ref_wire", text="", icon=icon)

        layout.operator("taichi.init_voronoi", icon="NODE_MATERIAL")
        if props.obj_gen:
            box = layout.box()
            box.label(text="Parameters", icon="SETTINGS")
            row = box.row(align=True)
            row.prop(props, "cells_shape", expand=True)
            box.prop(props, "cells_space")
            if props.cells_shape == "CUBIC":
                box.prop(props, "min_edge")
            box.prop(props, "resolution")

            row = box.row(align=True)
            row.prop(props, "boundary_type", expand=True)
            if props.boundary_type == "AUTO":
                box.prop(props, "clip_margin")
            elif props.boundary_type == "FIXED":
                row = box.row(align=True)
                row.prop(props, "boundary_width")
                row.prop(props, "boundary_height")

            box.label(text="Soap Bubble", icon="SPHERE")
            box.prop(props, "use_max_radius")
            if props.use_max_radius:
                box.prop(props, "max_radius")

            box = layout.box()
            box.label(text="Visuals", icon="RESTRICT_RENDER_OFF")
            box.prop(props, "bevel_radius")
            box.prop(props, "bevel_resolution")

            box = layout.box()
            box.label(text="Components", icon="OBJECT_DATA")
            row = box.row(align=True)
            row.operator("taichi.select_voronoi", text="Select Ref").target = "REF"
            row.operator("taichi.select_voronoi", text="Select Gen").target = "GEN"

            sec = layout.box()
            sec.label(text="Data Interface", icon="SPREADSHEET")
            col = sec.column(align=True)
            for names in [("Name", "Domain", "Type"), (ATTR_CELL_INDEX, "Point", "INT")]:
                row = col.row(align=True)
                b = row.box()
                b.scale_y = 0.5
                r = b.row(align=True)
                split = r.split(factor=0.6)
                split.label(text=names[0])
                split.label(text=names[1])


# =========================
# 6. Core Update Logic
# =========================


def update_voronoi(scene, depsgraph=None):
    props = scene.taichi_voronoi
    if not props.obj_ref or not props.obj_gen or not props.is_dirty:
        return

    setup_gn_viz(props.obj_gen, props)

    # Support modifiers/GN via vectorized extraction
    seeds_world = extract_seeds(props.obj_ref, depsgraph)
    if not seeds_world:
        return

    cells = extract_voronoi_cells(seeds_world)

    # Calculate organization clipping box based on mode
    org_min, org_max = None, None
    if props.boundary_type == "AUTO":
        seeds_2d = [s.xy for s in seeds_world]
        s_min = mathutils.Vector((min(p.x for p in seeds_2d), min(p.y for p in seeds_2d)))
        s_max = mathutils.Vector((max(p.x for p in seeds_2d), max(p.y for p in seeds_2d)))
        diag = (s_max - s_min).length
        margin = diag * props.clip_margin if diag > 0 else props.max_radius
        org_min = s_min - mathutils.Vector((margin, margin))
        org_max = s_max + mathutils.Vector((margin, margin))
    elif props.boundary_type == "FIXED":
        hw, hh = props.boundary_width * 0.5, props.boundary_height * 0.5
        org_min = mathutils.Vector((-hw, -hh))
        org_max = mathutils.Vector((hw, hh))

    segments = []
    cell_indices = []

    for c_idx, cell in enumerate(cells):
        seed_3d = cell["seed"]
        v_poly = cell["verts"]
        s_xy = seed_3d.xy

        # Step 1: Soap Bubble Radius (Local Clipping)
        if props.use_max_radius:
            r = props.max_radius
            v_poly = clip_poly_to_rect(v_poly, s_xy - mathutils.Vector((r, r)), s_xy + mathutils.Vector((r, r)))

        # Step 2: Organization Clipping (Stretching logic for smooth transition)
        if props.boundary_type != "NONE" and org_min and org_max:
            if not props.use_max_radius:
                # Strict mode: Seed must be inside to have any geometry
                if not (org_min.x <= s_xy.x <= org_max.x and org_min.y <= s_xy.y <= org_max.y):
                    continue
                v_poly = clip_poly_to_rect(v_poly, org_min, org_max)
            else:
                # Bubble mode: Stretched clipping box expanded to include the seed's own territory
                # This ensures that "inner" walls still clip the cell, but "outer" walls
                # are pushed away so the bubble can be round.
                r = props.max_radius
                
                # For each side, if the seed is outside, we push that side out
                stretched_min = mathutils.Vector((
                    min(org_min.x, s_xy.x - r),
                    min(org_min.y, s_xy.y - r)
                ))
                stretched_max = mathutils.Vector((
                    max(org_max.x, s_xy.x + r),
                    max(org_max.y, s_xy.y + r)
                ))
                
                v_poly = clip_poly_to_rect(v_poly, stretched_min, stretched_max)

        v_poly = retract_polygon(v_poly, s_xy, props.cells_space)
        n = len(v_poly)
        if n < 3:
            continue

        z = seed_3d.z

        if props.cells_shape == "EDGES":
            for i in range(n):
                v1, v2 = v_poly[i], v_poly[(i + 1) % n]
                v1_3d = mathutils.Vector((v1.x, v1.y, z))
                v2_3d = mathutils.Vector((v2.x, v2.y, z))
                segments.append((v1_3d, v1_3d, v2_3d, v2_3d))
                cell_indices.append(c_idx)
        elif props.cells_shape == "QUADRATIC":
            for i in range(n):
                v_curr = v_poly[i]
                v_next = v_poly[(i + 1) % n]
                v_prev = v_poly[(i - 1) % n]
                m_start = (v_prev + v_curr) * 0.5
                m_end = (v_curr + v_next) * 0.5
                s_3d = mathutils.Vector((m_start.x, m_start.y, z))
                e_3d = mathutils.Vector((m_end.x, m_end.y, z))
                c_3d = mathutils.Vector((v_curr.x, v_curr.y, z))
                segments.append((s_3d, c_3d, c_3d, e_3d))
                cell_indices.append(c_idx)
        elif props.cells_shape == "CUBIC":
            valid_indices = [i for i in range(n) if (v_poly[(i + 1) % n] - v_poly[i]).length > props.min_edge]
            nv = len(valid_indices)
            if nv < 2:
                continue
            for j in range(nv):
                idx_edge_curr = valid_indices[j]
                idx_edge_next = valid_indices[(j + 1) % nv]
                m_start = (v_poly[idx_edge_curr] + v_poly[(idx_edge_curr + 1) % n]) * 0.5
                m_end = (v_poly[idx_edge_next] + v_poly[(idx_edge_next + 1) % n]) * 0.5
                p1 = v_poly[(idx_edge_curr + 1) % n]
                p2 = v_poly[idx_edge_next]
                p0_3d = mathutils.Vector((m_start.x, m_start.y, z))
                p1_3d = mathutils.Vector((p1.x, p1.y, z))
                p2_3d = mathutils.Vector((p2.x, p2.y, z))
                p3_3d = mathutils.Vector((m_end.x, m_end.y, z))
                segments.append((p0_3d, p1_3d, p2_3d, p3_3d))
                cell_indices.append(c_idx)

    if not segments:
        props.obj_gen.data.clear_geometry()
        props.is_dirty = False
        return

    n_seg = len(segments)
    res = props.resolution
    n_v = n_seg * res
    p0_arr = np.array([s[0] for s in segments], dtype=np.float32)
    p1_arr = np.array([s[1] for s in segments], dtype=np.float32)
    p2_arr = np.array([s[2] for s in segments], dtype=np.float32)
    p3_arr = np.array([s[3] for s in segments], dtype=np.float32)
    ci_in = np.array(cell_indices, dtype=np.int32)
    v_out = np.zeros((n_v, 3), dtype=np.float32)
    ci_out = np.zeros(n_v, dtype=np.int32)
    sample_bezier_segments_kernel(p0_arr, p1_arr, p2_arr, p3_arr, res, v_out, ci_in, ci_out)

    e_idx = np.empty((n_seg * (res - 1) * 2), dtype=np.int32)
    for s in range(n_seg):
        base = s * res
        for i in range(res - 1):
            idx = (s * (res - 1) + i) * 2
            e_idx[idx], e_idx[idx + 1] = base + i, base + i + 1

    sync = sync_mesh_edit if props.obj_gen.mode == "EDIT" else sync_mesh_object
    sync(props.obj_gen.data, n_v, e_idx, v_out, ci_out)
    props.is_dirty = False


@bpy.app.handlers.persistent
def voronoi_handler(scene, depsgraph):
    props = scene.taichi_voronoi
    if props.obj_ref:
        for update in depsgraph.updates:
            if update.id.name in [props.obj_ref.name, getattr(props.obj_ref.data, "name", "")]:
                props.is_dirty = True
                break
    update_voronoi(scene, depsgraph)


classes = (
    TAI_VoronoiProperties,
    TAI_OT_init_voronoi,
    TAI_OT_toggle_ref_wire,
    TAI_OT_select_voronoi,
    TAI_PT_voronoi_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.taichi_voronoi = bpy.props.PointerProperty(type=TAI_VoronoiProperties)
    bpy.app.handlers.depsgraph_update_post.append(voronoi_handler)


def unregister():
    bpy.app.handlers.depsgraph_update_post[:] = [
        h for h in bpy.app.handlers.depsgraph_update_post if h.__name__ != "voronoi_handler"
    ]
    del bpy.types.Scene.taichi_voronoi
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
