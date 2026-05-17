import bpy
import bmesh
import numpy as np
import taichi as ti

import mathutils

mathutils.geometry.delaunay_2d_cdt

# =========================
# 1. Taichi Initialization
# =========================
if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)


@ti.kernel
def compute_fibonacci_points_kernel(n: int, is_hemisphere: int, verts_out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    """Compute Fibonacci sphere or hemisphere points."""
    golden_ratio = (1.0 + ti.math.sqrt(5.0)) / 2.0
    for i in range(n):
        z = 0.0
        if is_hemisphere == 1:
            # 0 to 1 range
            z = (float(i) + 0.5) / float(n)
        else:
            # -1 to 1 range
            z = 2.0 * (float(i) + 0.5) / float(n) - 1.0

        r2 = ti.math.clamp(1.0 - z * z, 0.0, 1.0)
        radius = ti.math.sqrt(r2)
        phi = 2.0 * ti.math.pi * float(i) / golden_ratio
        verts_out[i, 0] = radius * ti.math.cos(phi)
        verts_out[i, 1] = radius * ti.math.sin(phi)
        verts_out[i, 2] = z


# =========================
# 2. Blender Mesh Sync (Object & Edit Mode)
# =========================


def build_voronoi_dual(points, is_sphere=True):
    """CPU-side Voronoi generation via Convex Hull dual."""
    bm_hull = bmesh.new()
    for p in points:
        bm_hull.verts.new(p)

    # 1. Compute Convex Hull
    try:
        bmesh.ops.convex_hull(bm_hull, input=bm_hull.verts)
    except:
        bm_hull.free()
        return bmesh.new(), []

    # 2. Triangulate for consistent dual mapping
    bmesh.ops.triangulate(bm_hull, faces=bm_hull.faces)

    bm_dual = bmesh.new()
    face_to_vert = {}

    # Dual vertices are the face centers (normals for unit sphere) of the hull
    for f in bm_hull.faces:
        face_to_vert[f] = bm_dual.verts.new(f.normal)

    bm_dual.verts.ensure_lookup_table()

    seeds_out = []
    # Dual faces correspond to the original hull vertices
    for v in bm_hull.verts:
        if len(v.link_faces) < 3:
            continue

        # Order faces around vertex to form a valid polygon
        loop = v.link_loops[0]
        first_loop = loop
        dual_verts = []
        is_boundary = False
        while True:
            dual_verts.append(face_to_vert[loop.face])
            next_loop = loop.link_loop_prev.link_loop_radial_next
            if next_loop == loop.link_loop_prev:
                is_boundary = True
                break
            loop = next_loop
            if loop == first_loop:
                break

        if not is_boundary and len(dual_verts) >= 3:
            try:
                f_new = bm_dual.faces.new(dual_verts)
                # Ensure normal orientation points outward
                if f_new.normal.dot(f_new.calc_center_median()) < 0:
                    f_new.normal_flip()
                # Store the original seed coordinate for this face
                seeds_out.append(v.co.copy())
            except ValueError:
                pass

    bm_hull.free()

    if not is_sphere:
        # Use bisect to clean up the hemisphere boundary
        bmesh.ops.bisect_plane(
            bm_dual,
            geom=bm_dual.verts[:] + bm_dual.edges[:] + bm_dual.faces[:],
            plane_co=(0, 0, 0),
            plane_no=(0, 0, 1),
            clear_inner=True,  # Keeps Z >= 0
        )
    return bm_dual, seeds_out


def sync_mesh_object(mesh, verts, edges, faces, face_attrs=None):
    """High-speed update using foreach_set for Object Mode."""
    n_v, n_e, n_f = len(verts), len(edges), len(faces)

    # Structural update if counts mismatch
    if len(mesh.vertices) != n_v or len(mesh.edges) != n_e or len(mesh.polygons) != n_f:
        mesh.clear_geometry()
        mesh.from_pydata(verts.tolist(), edges, faces)
    else:
        # Fast update for coordinates only
        mesh.vertices.foreach_set("co", verts.flatten())

    # Handle face attributes (e.g., seed_pos)
    if face_attrs:
        for attr_name, data in face_attrs.items():
            if attr_name not in mesh.attributes:
                mesh.attributes.new(name=attr_name, type="FLOAT_VECTOR", domain="FACE")
            mesh.attributes[attr_name].data.foreach_set("vector", data.flatten())

    mesh.update()


def sync_mesh_edit(mesh_data, verts, edges, faces, face_attrs=None):
    """Surgical BMesh update for real-time Edit Mode feedback."""
    bm = bmesh.from_edit_mesh(mesh_data)
    n_v, n_e, n_f = len(verts), len(edges), len(faces)

    if len(bm.verts) != n_v or len(bm.edges) != n_e or len(bm.faces) != n_f:
        bm.clear()
        # Rebuild topology
        for v_co in verts:
            bm.verts.new(v_co)
        bm.verts.ensure_lookup_table()
        for e_idx in edges:
            try:
                bm.edges.new((bm.verts[e_idx[0]], bm.verts[e_idx[1]]))
            except:
                pass
        for f_idx in faces:
            try:
                bm.faces.new([bm.verts[i] for i in f_idx])
            except:
                pass
    else:
        # Fast update for coordinates
        bm.verts.ensure_lookup_table()
        for i, v in enumerate(bm.verts):
            v.co = verts[i]

    # Handle face attributes in BMesh
    if face_attrs:
        for attr_name, data in face_attrs.items():
            layer = bm.faces.layers.float_vector.get(attr_name) or bm.faces.layers.float_vector.new(attr_name)
            bm.faces.ensure_lookup_table()
            for i, f in enumerate(bm.faces):
                if i < len(data):
                    f[layer] = data[i]

    # CRITICAL: Recalculate normals to prevent "black faces" in viewport
    bm.normal_update()

    # Update the edit mesh with specific flags for a full viewport refresh
    bmesh.update_edit_mesh(mesh_data, loop_triangles=True)


def get_or_create_obj(name):
    """Helper to get or create a mesh object."""
    obj = bpy.data.objects.get(name)
    if not obj:
        mesh = bpy.data.meshes.new(name + "_Mesh")
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
    return obj


def update_fibonacci(self, context):
    """Callback to update the mesh when properties change."""
    props = context.scene.fibonacci_props
    n = props.n_points

    # 1. Taichi Computation
    verts_arr = np.zeros((n, 3), dtype=np.float32)
    is_hemi_int = 1 if props.geometry_mode == "HEMISPHERE" else 0
    compute_fibonacci_points_kernel(n, is_hemi_int, verts_arr)

    # 2. Update All Components
    targets = [
        ("Fibonacci_Lab", "NONE"),
        ("Fibonacci_Spiral", "SPIRAL"),
        ("Fibonacci_Voronoi", "VORONOI"),
    ]

    for obj_name, sub_fill in targets:
        obj = get_or_create_obj(obj_name)
        edges, faces, f_attrs = [], [], {}
        v_final = verts_arr

        if sub_fill == "VORONOI":
            if n < 4:
                v_final, e_final, f_final = verts_arr, [], []
            else:
                is_sphere = props.geometry_mode == "SPHERE"
                bm_voronoi, seeds = build_voronoi_dual(verts_arr, is_sphere=is_sphere)
                bm_voronoi.verts.index_update()
                bm_voronoi.edges.index_update()
                bm_voronoi.faces.index_update()
                v_final = np.array([v.co for v in bm_voronoi.verts], dtype=np.float32)
                e_final = [[v.index for v in e.verts] for e in bm_voronoi.edges]
                f_final = [[v.index for v in f.verts] for f in bm_voronoi.faces]
                if seeds:
                    f_attrs["seed_pos"] = np.array(seeds, dtype=np.float32)
                bm_voronoi.free()
        elif sub_fill == "SPIRAL" or sub_fill == "NONE":
            # Only compute edges for base lab if use_edges is enabled
            if sub_fill == "NONE" and not props.use_edges:
                edges = []
            else:
                c_mode = props.connection_mode
                if c_mode == "SEQUENTIAL":
                    edges = [(i, i + 1) for i in range(n - 1)]
                elif c_mode == "PHYLLOTAXIS":
                    s1, s2 = props.step_cw, props.step_ccw
                    if props.show_cw:
                        edges.extend([(i, i + s1) for i in range(n - s1)])
                    if props.show_ccw:
                        edges.extend([(i, i + s2) for i in range(n - s2)])
                    if sub_fill == "SPIRAL":
                        limit = n - (s1 + s2)
                        if limit > 0:
                            for i in range(limit):
                                faces.append((i, i + s1, i + s1 + s2, i + s2))
            v_final, e_final, f_final = verts_arr, edges, faces

        sync = sync_mesh_edit if obj.mode == "EDIT" else sync_mesh_object
        sync(obj.data, v_final, e_final, f_final, face_attrs=f_attrs)


# =========================
# 3. UI & Properties
# =========================


class TAI_OT_select_object(bpy.types.Operator):
    """Surgical selection of generated components."""

    bl_idname = "taichi.select_object"
    bl_label = "Select"
    bl_options = {"UNDO"}
    name: bpy.props.StringProperty()

    def execute(self, context):
        obj = bpy.data.objects.get(self.name)
        if obj:
            if context.object and context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            context.view_layer.objects.active = obj
        return {"FINISHED"}


class FibonacciProperties(bpy.types.PropertyGroup):
    geometry_mode: bpy.props.EnumProperty(
        name="Geometry",
        items=[
            ("SPHERE", "Full Sphere", "Complete Fibonacci sphere"),
            ("HEMISPHERE", "Hemisphere", "Upper half sphere only"),
        ],
        default="HEMISPHERE",
        update=update_fibonacci,
    )

    n_points: bpy.props.IntProperty(
        name="Points",
        description="Number of points on the Fibonacci sphere",
        default=500,
        min=2,
        max=50000,
        update=update_fibonacci,
    )

    connection_mode: bpy.props.EnumProperty(
        name="Connection",
        items=[
            ("NONE", "None", "No edges"),
            ("SEQUENTIAL", "Sequential", "Connect 0-1-2..."),
            ("PHYLLOTAXIS", "Phyllotaxis", "Spiral connections"),
        ],
        default="PHYLLOTAXIS",
        update=update_fibonacci,
    )

    use_edges: bpy.props.BoolProperty(
        name="Show Edges",
        description="Connect points with edges even when no faces are filled",
        default=True,
        update=update_fibonacci,
    )

    # Phyllotaxis specific
    show_cw: bpy.props.BoolProperty(name="Show Clockwise", default=True, update=update_fibonacci)
    step_cw: bpy.props.IntProperty(name="Step CW", default=21, min=1, update=update_fibonacci)

    show_ccw: bpy.props.BoolProperty(name="Show Counter-CW", default=True, update=update_fibonacci)
    step_ccw: bpy.props.IntProperty(name="Step CCW", default=13, min=1, update=update_fibonacci)


class VIEW3D_PT_fibonacci_lab(bpy.types.Panel):
    bl_label = "Fibonacci Lab"
    bl_idname = "VIEW3D_PT_fibonacci_lab"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout = self.layout
        props = context.scene.fibonacci_props

        layout.prop(props, "geometry_mode")
        layout.prop(props, "n_points", icon="DOT")
        layout.prop(props, "connection_mode", icon="EDGESEL")

        if props.connection_mode == "PHYLLOTAXIS":
            box = layout.box()
            col = box.column(align=True)
            row = col.row(align=True)
            row.prop(props, "show_cw", text="", icon="FORWARD")
            row.prop(props, "step_cw")

            row = col.row(align=True)
            row.prop(props, "show_ccw", text="", icon="BACK")
            row.prop(props, "step_ccw")

            col.label(text="Hint: Use Fibonacci numbers (13, 21, 34...)", icon="INFO")

        layout.prop(props, "use_edges", icon="MOD_WIREFRAME")

        layout.separator()

        # Components Section
        box = layout.box()
        box.label(text="Components", icon="OBJECT_DATA")
        col = box.column(align=True)

        comp_list = [
            ("Base Lab", "Fibonacci_Lab"),
            ("Spiral", "Fibonacci_Spiral"),
            ("Voronoi", "Fibonacci_Voronoi"),
        ]

        for label, name in comp_list:
            obj = bpy.data.objects.get(name)
            row = col.row(align=True)
            row.label(text=label)

            if obj:
                # Select
                op = row.operator(TAI_OT_select_object.bl_idname, text="", icon="RESTRICT_SELECT_OFF")
                op.name = name
                # Visibility (Viewport)
                row.prop(obj, "hide_viewport", text="", emboss=False)
                # Visibility (Render)
                row.prop(obj, "hide_render", text="", emboss=False)
            else:
                row.label(text=" -", icon="BLANK1")

        layout.separator()

        # Data Interface Disclosure (Mandatory Pattern)
        layout.label(text="Data Interface", icon="SPREADSHEET")
        col = layout.column(align=True)

        # Header
        row = col.row(align=True)
        box = row.box()
        box.scale_y = 0.5
        r = box.row(align=True)
        r.label(text="Attribute")
        r.label(text="Domain")

        # Position
        row = col.row(align=True)
        box = row.box()
        box.scale_y = 0.5
        r = box.row(align=True)
        r.label(text="position")
        r.label(text="Point")

        # Edges
        row = col.row(align=True)
        box = row.box()
        box.scale_y = 0.5
        r = box.row(align=True)
        r.label(text="edge")
        r.label(text="Edge")

        # Faces & seed_pos
        row = col.row(align=True)
        box = row.box()
        box.scale_y = 0.5
        r = box.row(align=True)
        r.label(text="face")
        r.label(text="Face")

        row = col.row(align=True)
        box = row.box()
        box.scale_y = 0.5
        r = box.row(align=True)
        r.label(text="seed_pos")
        r.label(text="Face")


# =========================
# 4. Registration
# =========================
classes = (
    FibonacciProperties,
    TAI_OT_select_object,
    VIEW3D_PT_fibonacci_lab,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.fibonacci_props = bpy.props.PointerProperty(type=FibonacciProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.fibonacci_props


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
    # Trigger initial update if object doesn't exist
    if not bpy.data.objects.get("Fibonacci_Lab"):
        update_fibonacci(None, bpy.context)
