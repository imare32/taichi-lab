"""
count=3;
a=(y,o=mag(k=cos(y*5)*(y<11?21:11),e=y/8-13)/6)=>point((q=k*2+49+cos(y)/k+k*cos(y/2)*(1+sin(o*4-e/2-t)))*sin(c=o/1.5-e/5-t/8+i%count*8)+200,230+q*cos(c)-79*sin(c/2))
t=0,draw=$=>{t||createCanvas(w=400,w);background(6).stroke(w,96);for(t+=PI/45,i=2e4;i--;)a(i/500)}
"""

import bpy
import numpy as np
import taichi as ti

# --- 1. Taichi Initialization ---
if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu)
    except:
        ti.init(arch=ti.cpu)


# --- 2. Taichi Kernel (保持原本优雅的数学公式) ---
@ti.kernel
def compute_fish_swarm(
    verts: ti.types.ndarray(dtype=ti.f32, ndim=2),
    fish_index: ti.types.ndarray(dtype=ti.f32, ndim=1),
    point_age: ti.types.ndarray(dtype=ti.f32, ndim=1),
    t: ti.f32,
    num_fishes: ti.i32,
    thickness_scale: ti.f32,
):
    for i in range(verts.shape[0]):
        # --- Identity Deconstruction ---
        f_id = ti.cast(i % num_fishes, ti.f32)
        y = ti.cast(i // num_fishes, ti.f32) / 500.0

        # --- Attribute Assignment ---
        fish_index[i] = f_id / ti.cast(num_fishes, ti.f32)
        point_age[i] = y / 20.0  # Normalized age

        # --- Mathematical Logic (p5.js original) ---
        k = ti.cos(y * 5.0) * (21.0 if y < 11.0 else 11.0)
        e = y / 8.0 - 13.0
        o = ti.sqrt(k * k + e * e) / 6.0

        # --- Original Jump Logic ---
        k_eps = 0.5
        inv_k = ti.cos(y) / (k if ti.abs(k) > k_eps else k_eps * (1.0 if k >= 0 else -1.0))
        inv_k = ti.math.clamp(inv_k, -0.1, 0.1)

        t_phase = t + f_id * 2.0

        q = k * 2.0 + 49.0 + inv_k + k * ti.cos(y / 2.0) * (1.0 + ti.sin(o * 4.0 - e / 2.0 - t_phase))
        c = o / 1.5 - e / 5.0 - t_phase / 8.0 + (f_id * 8.0)

        # --- 3D Expansion Math ---
        bulge = ti.sin(y * 3.1415926 / 40.0)
        wave = ti.sin(c * 1.5 + o * 2.0)
        y_val = bulge * wave * thickness_scale * 25.0

        # --- Spatial Mapping ---
        scale = 0.05
        spacing = 8.0
        x_offset = (f_id - (ti.cast(num_fishes, ti.f32) - 1.0) / 2.0) * spacing

        verts[i, 0] = (q * ti.sin(c)) * scale + x_offset
        verts[i, 1] = y_val * scale + (f_id * 0.1)
        verts[i, 2] = (q * ti.cos(c) - 79.0 * ti.sin(c / 2.0)) * scale


# --- 3. Update and Initialization Logic ---
def generate_topology(num_fishes, points_per_fish):
    """Generate edges topology list based on step logic"""
    edges = []
    for f in range(num_fishes):
        for p in range(points_per_fish - 1):
            # Connect adjacent sampling points in the same fish
            v1 = p * num_fishes + f
            v2 = (p + 1) * num_fishes + f
            edges.append((v1, v2))
    return edges


def update_fish_swarm(self, context):
    obj_name = context.scene.get("fish_swarm_obj", "")
    if not obj_name or obj_name not in bpy.data.objects:
        return

    obj = bpy.data.objects[obj_name]

    # Mode Safety Gating
    if obj.mode != "OBJECT":
        return

    mesh = obj.data

    num_fishes = context.scene.fish_count
    points_per_fish = 10000  # Restored to original detail level
    total_count = points_per_fish * num_fishes
    t = context.scene.fish_time
    thickness = context.scene.fish_thickness

    # 1. Initialize Topology (if vertex count changes)
    if len(mesh.vertices) != total_count:
        mesh.clear_geometry()
        verts_init = np.zeros((total_count, 3), dtype=np.float32)
        edges = generate_topology(num_fishes, points_per_fish)
        # Write topology with Edges
        mesh.from_pydata(verts_init.tolist(), edges, [])

        # Initialize Attributes
        if not mesh.attributes.get("fish_index"):
            mesh.attributes.new(name="fish_index", type="FLOAT", domain="POINT")
        if not mesh.attributes.get("point_age"):
            mesh.attributes.new(name="point_age", type="FLOAT", domain="POINT")

        # Display settings
        obj.display_type = "WIRE"

    # 2. Prepare Taichi Output
    verts_arr = np.zeros((total_count, 3), dtype=np.float32)
    fish_idx_arr = np.zeros(total_count, dtype=np.float32)
    point_age_arr = np.zeros(total_count, dtype=np.float32)

    compute_fish_swarm(verts_arr, fish_idx_arr, point_age_arr, t, num_fishes, thickness)

    # 3. Fast coordinate and attribute updates
    mesh.vertices.foreach_set("co", verts_arr.flatten())

    # Sync Attributes
    mesh.attributes["fish_index"].data.foreach_set("value", fish_idx_arr)
    mesh.attributes["point_age"].data.foreach_set("value", point_age_arr)

    mesh.update()


# --- 4. Operator and UI ---
class TAI_OT_create_fish_swarm(bpy.types.Operator):
    """Create a 3D fish swarm with linear topology"""

    bl_idname = "taichi.create_fish_swarm"
    bl_label = "Create Fish Swarm"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mesh = bpy.data.meshes.new("FishSwarmMesh")
        obj = bpy.data.objects.new("FishSwarm", mesh)
        context.collection.objects.link(obj)
        context.scene["fish_swarm_obj"] = obj.name
        context.scene.fish_animate = True
        update_fish_swarm(self, context)
        return {"FINISHED"}


class TAI_PT_fish_swarm_panel(bpy.types.Panel):
    bl_label = "Taichi Fish Swarm"
    bl_idname = "TAI_PT_fish_swarm_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Taichi"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        col = layout.column(align=True)
        col.operator(TAI_OT_create_fish_swarm.bl_idname, icon="PARTICLE_DATA")

        layout.separator()

        box = layout.box()
        box.label(text="Simulation Parameters", icon="PHYSICS")
        box.prop(scene, "fish_count")
        box.prop(scene, "fish_thickness")
        box.prop(scene, "fish_time")
        box.prop(scene, "fish_animate", toggle=True, icon="PLAY" if scene.fish_animate else "PAUSE")

        # Data Interface Disclosure
        layout.separator()
        layout.label(text="Data Interface", icon='SPREADSHEET')

        col = layout.column(align=True)

        # Attribute: fish_index
        row = col.row(align=True)
        box = row.box()
        box.scale_y = 0.5
        r = box.row(align=True)
        r.label(text="fish_index")
        r.label(text="Point")

        # Attribute: point_age
        row = col.row(align=True)
        box = row.box()
        box.scale_y = 0.5
        r = box.row(align=True)
        r.label(text="point_age")
        r.label(text="Point")



# --- 5. Registration ---
@bpy.app.handlers.persistent
def fish_swarm_animation_callback(scene):
    # Reset time when returning to frame 1
    if scene.frame_current <= 1:
        scene.fish_time = 0.0

    if hasattr(scene, "fish_animate") and scene.fish_animate:
        scene.fish_time += 0.05


classes = (
    TAI_OT_create_fish_swarm,
    TAI_PT_fish_swarm_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.fish_count = bpy.props.IntProperty(
        name="Fish Count", default=3, min=1, max=50, update=update_fish_swarm
    )
    bpy.types.Scene.fish_thickness = bpy.props.FloatProperty(
        name="Thickness", default=1.0, min=0.0, max=5.0, update=update_fish_swarm
    )
    bpy.types.Scene.fish_time = bpy.props.FloatProperty(name="Time", default=0.0, update=update_fish_swarm)
    bpy.types.Scene.fish_animate = bpy.props.BoolProperty(name="Animate", default=True)

    # Surgical cleanup and registration of handlers
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre if h.__name__ != fish_swarm_animation_callback.__name__
    ]
    bpy.app.handlers.frame_change_pre.append(fish_swarm_animation_callback)


def unregister():
    # Remove properties
    del bpy.types.Scene.fish_count
    del bpy.types.Scene.fish_thickness
    del bpy.types.Scene.fish_time
    del bpy.types.Scene.fish_animate

    # Remove handlers
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre if h.__name__ != fish_swarm_animation_callback.__name__
    ]

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
