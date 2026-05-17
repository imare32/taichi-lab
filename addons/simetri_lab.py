import gc
import itertools

import bmesh
import bpy
import simetri.graphics as sg

# --- Core Specifications & Naming ---
REF_OBJ_NAME = "Simetri_Ref"
GEN_OBJ_NAME = "Simetri_Lace"

# --- Utility Functions ---


def convert_coord(coord_2d):
    return (coord_2d[0] / 200, coord_2d[1] / 200, 0)


def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function"""

    class K(object):
        __slots__ = ["obj"]

        def __init__(self, obj):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        __hash__ = None

    return K


def group_into_bins(values, delta, compare_function=None, with_objects=True):
    if not values:
        return []
    if compare_function:
        values = sorted(values, key=cmp_to_key(compare_function))
    else:
        values = sorted(values)
    bins = []
    bin_ = [values[0]]
    if with_objects:
        for value in values[1:]:
            if value[0] - bin_[0][0] <= delta:
                bin_.append(value)
            else:
                bins.append(bin_)
                bin_ = [value]
        bins.append(bin_)
    else:
        for value in values[1:]:
            if value - bin_[0] <= delta:
                bin_.append(value)
            else:
                bins.append(bin_)
                bin_ = [value]
        bins.append(bin_)
    return bins


def fill_fragments(fragments):
    if not fragments:
        return
    areas = []
    for fragment in fragments:
        areas.append((fragment.area, fragment.id))
    bins = group_into_bins(areas, 2)

    palette = sg.swatches[1]
    n_palette = len(palette)
    n_bins = len(bins)
    if n_bins == 0:
        return

    palette = palette[:: max(1, n_palette // n_bins)]
    palette = [sg.Color(*c) for c in palette]
    n = len(palette)
    for i, bin_ in enumerate(bins):
        color = palette[i % n]
        for _, fragment_id in bin_:
            fragment = sg.d_id_obj.get(fragment_id)
            if fragment:
                fragment.color = color


# --- Simulation & Sync Logic ---


def blender_to_simetri(obj, depsgraph=None):
    """Extracts continuous edge loops from the evaluated Blender mesh."""
    if not obj or obj.type != "MESH":
        return None

    # If no depsgraph provided (e.g. manual update), get the current one
    if depsgraph is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()

    # Get the evaluated object to capture modifiers
    eval_obj = obj.evaluated_get(depsgraph)

    # In Edit Mode, obj.data doesn't reflect live edits.
    # We must read from the edit bmesh instead, but only copy —
    # never modify the live edit bmesh (remove_doubles would corrupt it).
    if obj.mode == "EDIT":
        edit_bm = bmesh.from_edit_mesh(obj.data)
        if edit_bm is None:
            return None
        bm = edit_bm.copy()
    else:
        bm = bmesh.new()
        bm.from_mesh(eval_obj.data)

    try:
        # Ensure topology is clean for tracing
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        shapes = []
        visited_edges = set()

        # Trace connected edges into paths
        for edge in bm.edges:
            if edge in visited_edges:
                continue

            path_verts = [edge.verts[0], edge.verts[1]]
            visited_edges.add(edge)

            # Trace forward
            v = path_verts[-1]
            while True:
                next_edge = None
                for e in v.link_edges:
                    if e not in visited_edges:
                        next_edge = e
                        break
                if next_edge:
                    visited_edges.add(next_edge)
                    v = next_edge.other_vert(v)
                    path_verts.append(v)
                    if v == path_verts[0]:
                        break
                else:
                    break

            # Trace backward (if not a cycle)
            if path_verts[0] != path_verts[-1]:
                v = path_verts[0]
                while True:
                    next_edge = None
                    for e in v.link_edges:
                        if e not in visited_edges:
                            next_edge = e
                            break
                    if next_edge:
                        visited_edges.add(next_edge)
                        v = next_edge.other_vert(v)
                        path_verts.insert(0, v)
                    else:
                        break

            if len(path_verts) > 1:
                # Simetri works in XY plane. Local coordinates * 200.
                coords = [(pv.co.x * 200, pv.co.y * 200) for pv in path_verts]
                is_closed = path_verts[0] == path_verts[-1]
                if is_closed:
                    if len(coords) > 1 and coords[0] == coords[-1]:
                        coords.pop()

                shape = sg.Shape(coords)
                shape.closed = is_closed
                shapes.append(shape)

        return sg.Batch(shapes)
    finally:
        bm.free()


def sync_simetri_mesh(obj, shapes):
    """Efficiently updates a Blender mesh object with Simetri shape data."""
    if not obj or obj.type != "MESH":
        return

    mesh = obj.data
    verts = []
    edges = []
    faces = []
    colors = []
    vid = 0

    for shape in shapes:
        verts.extend([(coord[0] / 200, coord[1] / 200, 0) for coord in shape])
        n_v = len(shape)
        edges.extend([(vid + i, vid + i + 1) for i in range(n_v - 1)])
        if shape.closed:
            edges.append((vid, vid + n_v - 1))

        if shape.is_polygon:
            faces.append([vid + i for i in range(n_v)])
            c = shape.color.rgba if hasattr(shape, "color") else (1.0, 1.0, 1.0, 1.0)
            colors.extend([c] * n_v)

        vid += n_v

    # Always rebuild geometry — the old "fast path" only checked vertex/edge/face
    # counts, not actual connectivity. Same counts but different topology caused
    # corrupted faces and misaligned vertex colors.
    mesh.clear_geometry()
    mesh.from_pydata(verts, edges, faces)
    if "color" not in mesh.attributes:
        mesh.attributes.new(name="color", type="FLOAT_COLOR", domain="POINT")

    # Update Vertex Colors (Attributes)
    if colors and "color" in mesh.attributes:
        flat_colors = list(itertools.chain.from_iterable(colors))
        mesh.attributes["color"].data.foreach_set("color", flat_colors)

    mesh.update()


def update_lace_process(scene, depsgraph=None):
    props = scene.simetri_props

    # 1. Source Data Retrieval
    if props.obj_ref:
        base_shapes = blender_to_simetri(props.obj_ref, depsgraph=depsgraph)
        if not base_shapes or not base_shapes.all_shapes:
            base_shapes = sg.stars.Star(7).level(3)
    else:
        base_shapes = sg.stars.Star(7).level(3)

    if not base_shapes:
        return

    # 2. Simulation Step
    try:
        lace = sg.Lace(base_shapes, offset=props.offset, fast_mode=True)
        fill_fragments(lace.fragments)
        display_shapes = lace.fragments + lace.plaits
    except Exception as e:
        import traceback
        import sys

        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = traceback.extract_tb(exc_traceback)
        print(f"Error Type: {exc_type.__name__}")
        print(f"Error Message: {exc_value}")
        print("\nTraceback:")
        for tb in traceback_details:
            print(f'  File: "{tb.filename}", line {tb.lineno}, Function: {tb.name}')
            if tb.line:
                print(f"    Code: {tb.line}")
        return

    # 3. Target Synchronization
    obj_gen = props.obj_gen
    if not obj_gen or obj_gen.name not in bpy.data.objects:
        mesh = bpy.data.meshes.new("Simetri_Lace_Mesh")
        obj_gen = bpy.data.objects.new(GEN_OBJ_NAME, mesh)
        bpy.context.collection.objects.link(obj_gen)
        props.obj_gen = obj_gen

    sync_simetri_mesh(obj_gen, display_shapes)


# --- Properties & UI ---


class SIMETRI_Props(bpy.types.PropertyGroup):
    obj_ref: bpy.props.PointerProperty(
        name="Reference Object",
        type=bpy.types.Object,
        description="Source object (Star(7) used if empty)",
        update=lambda self, context: self.set_dirty(),
    )
    obj_gen: bpy.props.PointerProperty(name="Lace Object", type=bpy.types.Object)
    offset: bpy.props.FloatProperty(
        name="Offset", default=5.0, min=0.1, soft_max=20.0, update=lambda self, context: self.set_dirty()
    )
    live_update: bpy.props.BoolProperty(name="Live Update", default=False)
    is_dirty: bpy.props.BoolProperty(default=True)

    def set_dirty(self):
        self.is_dirty = True


class SIMETRI_PT_Main(bpy.types.Panel):
    bl_label = "Simetri Lab"
    bl_idname = "SIMETRI_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Simetri"

    def draw(self, context):
        layout = self.layout
        props = context.scene.simetri_props

        col = layout.column(align=True)
        col.prop(props, "obj_ref")
        col.prop(props, "offset")

        row = layout.row(align=True)
        row.prop(props, "live_update", toggle=True, icon="PLAY" if props.live_update else "PAUSE")
        if not props.live_update:
            row.operator("simetri.update", icon="FILE_REFRESH", text="Update")

        layout.operator("simetri.setup", icon="PRESET")

        layout.separator()
        layout.prop(props, "obj_gen")

        # Data Interface Disclosure
        layout.label(text="Data Interface", icon="SPREADSHEET")
        box = layout.box()
        box.scale_y = 0.5
        r = box.row(align=True)
        r.label(text="color")
        r.label(text="POINT")


class SIMETRI_OT_Update(bpy.types.Operator):
    bl_label = "Update Lace"
    bl_idname = "simetri.update"

    def execute(self, context):
        update_lace_process(context.scene, depsgraph=context.evaluated_depsgraph_get())
        context.scene.simetri_props.is_dirty = False
        return {"FINISHED"}


class SIMETRI_OT_Setup(bpy.types.Operator):
    """Initializes the lab environment with default objects"""

    bl_label = "Setup Lab"
    bl_idname = "simetri.setup"

    def execute(self, context):
        props = context.scene.simetri_props

        # Create Reference if missing
        if not props.obj_ref:
            mesh = bpy.data.meshes.new("Simetri_Ref_Mesh")
            obj = bpy.data.objects.new(REF_OBJ_NAME, mesh)
            context.collection.objects.link(obj)
            props.obj_ref = obj

            star_batch = sg.stars.Star(7).level(3)
            sync_simetri_mesh(obj, star_batch.all_shapes)

        update_lace_process(context.scene, depsgraph=context.evaluated_depsgraph_get())

        # Viewport setup
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.color_type = "VERTEX"

        return {"FINISHED"}


# --- Handlers ---


@bpy.app.handlers.persistent
def simetri_deps_handler(scene, depsgraph):
    props = getattr(scene, "simetri_props", None)
    if not props or not props.live_update:
        return

    should_update = props.is_dirty

    if not should_update and props.obj_ref:
        for update in depsgraph.updates:
            if update.id.name in [props.obj_ref.name, getattr(props.obj_ref.data, "name", "")]:
                should_update = True
                break

    if should_update:
        gc.disable()
        try:
            update_lace_process(scene, depsgraph=depsgraph)
            props.is_dirty = False
        except Exception as e:
            print(f"Handler Error: {e}")
        finally:
            gc.enable()


def purge_handlers():
    bpy.app.handlers.depsgraph_update_post[:] = [
        h for h in bpy.app.handlers.depsgraph_update_post if h.__name__ != simetri_deps_handler.__name__
    ]


# --- Lifecycle ---

classes = [
    SIMETRI_Props,
    SIMETRI_PT_Main,
    SIMETRI_OT_Update,
    SIMETRI_OT_Setup,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.simetri_props = bpy.props.PointerProperty(type=SIMETRI_Props)

    purge_handlers()
    bpy.app.handlers.depsgraph_update_post.append(simetri_deps_handler)


def unregister():
    purge_handlers()
    if hasattr(bpy.types.Scene, "simetri_props"):
        del bpy.types.Scene.simetri_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
