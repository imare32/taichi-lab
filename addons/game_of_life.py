import bmesh
import bpy
import numpy as np
import taichi as ti
import random

# =========================
# 0. Global Constants & Taichi Init
# =========================

ATTR_ALIVE = "alive"
MAX_NEIGHBORS = 12

if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu)
    except Exception:
        ti.init(arch=ti.cpu)

# =========================
# 1. Taichi Kernels (Parallel Computation)
# =========================

@ti.kernel
def compute_next_generation_kernel(
    current: ti.types.ndarray(dtype=ti.f32, ndim=1),
    target: ti.types.ndarray(dtype=ti.f32, ndim=1),
    neighbors: ti.types.ndarray(dtype=ti.i32, ndim=2),
    neighbor_counts: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    """Apply Conway's Game of Life rules in parallel."""
    for i in range(current.shape[0]):
        alive_neighbors = 0.0
        count = neighbor_counts[i]
        for j in range(count):
            nb_idx = neighbors[i, j]
            if current[nb_idx] > 0.5:
                alive_neighbors += 1.0
        
        # Conway's Rules
        if current[i] > 0.5:
            # Any live cell with two or three live neighbours survives.
            if alive_neighbors == 2.0 or alive_neighbors == 3.0:
                target[i] = 1.0
            else:
                target[i] = 0.0
        else:
            # Any dead cell with exactly three live neighbours becomes a live cell.
            if alive_neighbors == 3.0:
                target[i] = 1.0
            else:
                target[i] = 0.0

@ti.kernel
def interpolate_states_kernel(
    current: ti.types.ndarray(dtype=ti.f32, ndim=1),
    target: ti.types.ndarray(dtype=ti.f32, ndim=1),
    display: ti.types.ndarray(dtype=ti.f32, ndim=1),
    t: float,
    mode: int,
):
    """Interpolate between generations for smooth transitions."""
    val_t = t
    if mode == 1: # Smoothstep
        val_t = t * t * (3.0 - 2.0 * t)
    elif mode == 2: # Ease-In
        val_t = t * t
    elif mode == 3: # Ease-Out
        val_t = 1.0 - (1.0 - t) * (1.0 - t)
        
    for i in range(current.shape[0]):
        display[i] = current[i] + (target[i] - current[i]) * val_t

# =========================
# 2. Geometry & Topology Utilities
# =========================

def extract_mesh_topology(obj_ref: bpy.types.Object):
    """Extract face-to-face neighbor mapping (Moore neighborhood)."""
    if obj_ref.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(obj_ref.data)
    else:
        bm = bmesh.new()
        bm.from_mesh(obj_ref.data)
    
    bm.faces.ensure_lookup_table()
    n_faces = len(bm.faces)
    
    neighbors = np.full((n_faces, MAX_NEIGHBORS), -1, dtype=np.int32)
    neighbor_counts = np.zeros(n_faces, dtype=np.int32)
    
    for face in bm.faces:
        idx = face.index
        nb_set = set()
        # Find all faces sharing at least one vertex
        for vert in face.verts:
            for linked_face in vert.link_faces:
                if linked_face != face:
                    nb_set.add(linked_face.index)
        
        # Clamp to MAX_NEIGHBORS
        nb_list = list(nb_set)[:MAX_NEIGHBORS]
        count = len(nb_list)
        neighbor_counts[idx] = count
        neighbors[idx, :count] = nb_list

    if obj_ref.mode != 'EDIT':
        bm.free()
        
    return neighbors, neighbor_counts

def setup_data_interface(mesh):
    """Ensure the 'alive' attribute exists on the face domain."""
    if ATTR_ALIVE not in mesh.attributes:
        mesh.attributes.new(name=ATTR_ALIVE, type='FLOAT', domain='FACE')

# =========================
# 3. Blender Mesh Sync
# =========================

def sync_mesh_object(mesh, data):
    """Fast attribute update in Object Mode."""
    if ATTR_ALIVE not in mesh.attributes:
        mesh.attributes.new(name=ATTR_ALIVE, type='FLOAT', domain='FACE')
    mesh.attributes[ATTR_ALIVE].data.foreach_set("value", data)
    # Note: mesh.update() is usually not needed for just attribute changes in 3.x+
    # but we trigger a depsgraph update if needed via the handler.

def sync_mesh_edit(mesh_data, data):
    """Surgical BMesh update for Edit Mode."""
    bm = bmesh.from_edit_mesh(mesh_data)
    layer = bm.faces.layers.float.get(ATTR_ALIVE) or bm.faces.layers.float.new(ATTR_ALIVE)
    
    bm.faces.ensure_lookup_table()
    for i, face in enumerate(bm.faces):
        if i < len(data):
            face[layer] = data[i]
            
    bmesh.update_edit_mesh(mesh_data)

# =========================
# 4. State & Properties
# =========================

class TAI_GOLProperties(bpy.types.PropertyGroup):
    obj_ref: bpy.props.PointerProperty(
        name="Reference", 
        type=bpy.types.Object, 
        poll=lambda s, o: o.type == 'MESH',
        description="Mesh must be a manifold quad mesh for best results"
    )
    
    # Simulation Parameters
    seed: bpy.props.IntProperty(name="Seed", default=42, min=0)
    density: bpy.props.FloatProperty(name="Initial Density", default=0.3, min=0.0, max=1.0)
    
    # Timeline Parameters
    frames_per_gen: bpy.props.IntProperty(name="Frames / Gen", default=10, min=1)
    transition_frames: bpy.props.IntProperty(name="Transition Frames", default=5, min=0)
    interpolation_mode: bpy.props.EnumProperty(
        name="Interpolation",
        items=[
            ('LINEAR', "Linear", ""),
            ('SMOOTHSTEP', "Smoothstep", ""),
            ('EASE_IN', "Ease In", ""),
            ('EASE_OUT', "Ease Out", ""),
        ],
        default='SMOOTHSTEP'
    )
    
    # Internal State (not exposed to UI directly)
    is_running: bpy.props.BoolProperty(default=False)
    last_tick_frame: bpy.props.IntProperty(default=-1)
    
    # Simulation Data (cached in Scene to survive handler calls)
    # These are handled via global dictionaries in this script for speed/simplicity
    # but could be stored in the property group if serialized.

# Global simulation state storage
GOL_STATE = {
    "current": None, # np.array
    "target": None,  # np.array
    "display": None, # np.array
    "neighbors": None,
    "nb_counts": None,
    "initialized": False
}

# =========================
# 5. UI Panels & Operators
# =========================

class TAI_OT_init_gol(bpy.types.Operator):
    bl_idname = "taichi.init_gol"
    bl_label = "Initialize Game of Life"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.taichi_gol
        
        # 0. Default Reference Generation
        if not props.obj_ref:
            bpy.ops.mesh.primitive_torus_add(major_radius=1, minor_radius=0.25, major_segments=48, minor_segments=12)
            props.obj_ref = context.active_object
            props.obj_ref.name = "GOL_Torus"
            
        obj = props.obj_ref
        
        # 1. Topology Extraction
        neighbors, nb_counts = extract_mesh_topology(obj)
        n_faces = len(nb_counts)
        
        # 2. State Initialization
        random.seed(props.seed)
        current = np.array([1.0 if random.random() < props.density else 0.0 for _ in range(n_faces)], dtype=np.float32)
        target = current.copy()
        display = current.copy()
        
        GOL_STATE["current"] = current
        GOL_STATE["target"] = target
        GOL_STATE["display"] = display
        GOL_STATE["neighbors"] = neighbors
        GOL_STATE["nb_counts"] = nb_counts
        GOL_STATE["initialized"] = True
        
        props.last_tick_frame = context.scene.frame_current
        props.is_running = True
        
        # 3. Initial Sync
        setup_data_interface(obj.data)
        sync = sync_mesh_edit if obj.mode == 'EDIT' else sync_mesh_object
        sync(obj.data, display)
        
        return {'FINISHED'}

class TAI_PT_gol_panel(bpy.types.Panel):
    bl_label = "Taichi Game of Life"
    bl_idname = "TAI_PT_gol_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Taichi'

    def draw(self, context):
        layout = self.layout
        props = context.scene.taichi_gol
        
        layout.prop(props, "obj_ref")
        layout.operator(TAI_OT_init_gol.bl_idname, icon='PLAY')
        
        if GOL_STATE["initialized"]:
            box = layout.box()
            box.label(text="Simulation Settings", icon='SETTINGS')
            box.prop(props, "seed")
            box.prop(props, "density")
            
            box = layout.box()
            box.label(text="Timeline Control", icon='TIME')
            box.prop(props, "frames_per_gen")
            box.prop(props, "transition_frames")
            box.prop(props, "interpolation_mode")
            
            # Data Interface Table
            sec = layout.box()
            sec.label(text="Data Interface", icon='SPREADSHEET')
            col = sec.column(align=True)
            
            # Header
            row = col.row(align=True)
            for text in ["Name", "Domain", "Type"]:
                b = row.box()
                b.scale_y = 0.75
                b.label(text=text)
            
            # Row
            row = col.row(align=True)
            for text in [ATTR_ALIVE, "Face", "FLOAT"]:
                b = row.box()
                b.scale_y = 0.5
                b.label(text=text)

# =========================
# 6. Core Update Logic (Handler)
# =========================

@bpy.app.handlers.persistent
def gol_frame_handler(scene, depsgraph):
    props = scene.taichi_gol
    if not props.is_running or not GOL_STATE["initialized"]:
        return
        
    obj = props.obj_ref
    if not obj:
        return

    n_faces = len(GOL_STATE["current"])
    frame = scene.frame_current
    
    # Calculate generation timing
    gen_frames = props.frames_per_gen
    trans_frames = min(props.transition_frames, gen_frames)
    
    # Determine progress within current generation cycle
    # We use a base frame to keep simulation consistent on jump/reverse
    cycle_frame = frame % gen_frames
    
    # Tick logic: When cycle_frame is 0, we compute the NEXT generation target
    # and set current state to the PREVIOUS target.
    # However, to avoid jumping on every frame change, we check if we actually transitioned.
    
    current_gen_idx = frame // gen_frames
    last_gen_idx = props.last_tick_frame // gen_frames
    
    if current_gen_idx != last_gen_idx:
        # Step the simulation
        GOL_STATE["current"][:] = GOL_STATE["target"]
        
        compute_next_generation_kernel(
            GOL_STATE["current"],
            GOL_STATE["target"],
            GOL_STATE["neighbors"],
            GOL_STATE["nb_counts"]
        )
        props.last_tick_frame = frame

    # Interpolation logic
    t = 1.0
    if trans_frames > 0:
        t = min(1.0, cycle_frame / trans_frames)
    
    mode_map = {'LINEAR': 0, 'SMOOTHSTEP': 1, 'EASE_IN': 2, 'EASE_OUT': 3}
    mode_idx = mode_map.get(props.interpolation_mode, 1)
    
    interpolate_states_kernel(
        GOL_STATE["current"],
        GOL_STATE["target"],
        GOL_STATE["display"],
        t,
        mode_idx
    )
    
    # Sync to Blender
    sync = sync_mesh_edit if obj.mode == 'EDIT' else sync_mesh_object
    sync(obj.data, GOL_STATE["display"])

# =========================
# 7. Registration
# =========================

classes = (
    TAI_GOLProperties,
    TAI_OT_init_gol,
    TAI_PT_gol_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.taichi_gol = bpy.props.PointerProperty(type=TAI_GOLProperties)
    
    # Clean up and register handler
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre if h.__name__ != gol_frame_handler.__name__
    ]
    bpy.app.handlers.frame_change_pre.append(gol_frame_handler)

def unregister():
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre if h.__name__ != gol_frame_handler.__name__
    ]
    del bpy.types.Scene.taichi_gol
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__" or __name__ == "<run_path>":
    register()
