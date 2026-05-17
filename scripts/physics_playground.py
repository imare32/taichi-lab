import time

import bpy
import gpu
import numpy as np
import taichi as ti
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Color

# --- 1. CONSTANTS ---
SIM_NAME = "Physics_Playground"
GN_GROUP_NAME = "PhysicsPlaygroundGN"
HANDLE_KEY = "physics_playground_draw_handle"
MAX_PARTICLES = 10000
DIM = 2
BOUNDARY = 5.0

SWATCHES = [
    ["#9F2158", "#782255", "#A58490", "#ECD1BF", "#BB5548", "#007D8E", "#6B5146", "#4B1A47", "#DCD3B2"],
    ["#E94709", "#27120A", "#C3AAB0", "#F7D5B3", "#8F8FB0", "#A593AD", "#F4B3C2", "#7B6667", "#4D3E4B"],
    ["#D2BE6C", "#A78B5B", "#E2D7A8", "#8C6D5C", "#767676", "#949495", "#A99E93", "#E5DECE", "#EAEBEB"],
    ["#839A5C", "#31673F", "#4B2D16", "#B6BB7E", "#C7B183", "#917347", "#CACCD2", "#F2E3D6", "#746D60"],
    ["#BC6238", "#FFFFFE", "#AFAFB0", "#2F2725", "#8A3618", "#34715F", "#005878", "#BA9146", "#4B5A69"],
    ["#4E67B0", "#18448E", "#0B1644", "#887B92", "#A59ACA", "#EEE3D9", "#E94E66", "#C085B8", "#DEC824"],
    ["#D7003A", "#674196", "#615C66", "#130012", "#F18D00", "#F9F8F8", "#EC6800", "#FAD5D6", "#F6AD48"],
    ["#E83828", "#67BE8D", "#F8B500", "#FCD475", "#FBFAF3", "#AED3ED", "#CF0060", "#007C45", "#2A2120"],
    ["#895687", "#CC7DB1", "#734E95", "#D2B74E", "#F8EED1", "#E6E6E6", "#EEC9D0", "#89C3EB", "#2A3322"],
    ["#F8CEDA", "#D1BADA", "#A0D8EF", "#68B7A1", "#D8E698", "#F0C1C0", "#E95295", "#7054A0", "#239DDA"],
]

# --- 2. TAICHI CORE ---
if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu)
    except:
        ti.init(arch=ti.cpu)

pos = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_PARTICLES)
vel = ti.Vector.field(DIM, dtype=ti.f32, shape=MAX_PARTICLES)
active_count = ti.field(dtype=ti.i32, shape=())

gravity_val = ti.field(dtype=ti.f32, shape=())
bounce_att = ti.field(dtype=ti.f32, shape=())
rand_x_vel = ti.field(dtype=ti.i32, shape=())

_coords_cache = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)


@ti.kernel
def init_sim():
    active_count[None] = 1
    for i in range(MAX_PARTICLES):
        pos[i], vel[i] = [0.0, 0.0], [0.0, 0.0]
    vel[0][0] = (ti.random() - 0.5) * 40.0


@ti.kernel
def spawn_particle_at(px: ti.f32, pz: ti.f32):
    if active_count[None] < MAX_PARTICLES:
        idx = active_count[None]
        pos[idx] = [px, pz]
        vx = (ti.random() - 0.5) * 40.0 if rand_x_vel[None] == 1 else 0.0
        vel[idx] = [vx, 0.0]
        active_count[None] += 1


@ti.kernel
def step_sim(dt: ti.f32, substeps: ti.i32):
    n, g, att = active_count[None], gravity_val[None], bounce_att[None]
    for _ in range(substeps):
        for i in range(n):
            vel[i][1] += g * dt
            pos[i] += vel[i] * dt
            for d in ti.static(range(DIM)):
                if abs(pos[i][d]) > BOUNDARY:
                    pos[i][d] = BOUNDARY if pos[i][d] > 0 else -BOUNDARY
                    vel[i][d] *= -att


# --- 3. UTILS ---
def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def update_color_ramp(group, swatch_idx):
    ramp = next((n for n in group.nodes if n.type == "VALTORGB"), None)
    if not ramp:
        return
    colors = SWATCHES[int(swatch_idx)]
    els = ramp.color_ramp.elements
    while len(els) < len(colors):
        els.new(0)
    while len(els) > len(colors):
        els.remove(els[-1])
    for i, hex_c in enumerate(colors):
        els[i].position = i / (len(colors) - 1) if len(colors) > 1 else 0
        rgb = hex_to_rgb(hex_c)
        els[i].color = (*Color(rgb).from_srgb_to_scene_linear(), 1.0)


# --- 4. BLENDER DATA MANAGEMENT ---
def get_sim_resources():
    obj = bpy.data.objects.get(SIM_NAME)
    if not obj:
        mesh = bpy.data.meshes.new(SIM_NAME)
        obj = bpy.data.objects.new(SIM_NAME, mesh)
        bpy.context.collection.objects.link(obj)
        mesh.from_pydata([(0, 0, 0)], [], [])

    group = bpy.data.node_groups.get(GN_GROUP_NAME)
    if not group:
        group = bpy.data.node_groups.new(name=GN_GROUP_NAME, type="GeometryNodeTree")
        group.is_modifier = True
        group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        n, links = group.nodes, group.links

        inp, out = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
        inp.location, out.location = (-700, 200), (900, 100)
        circle = n.new("GeometryNodeCurvePrimitiveCircle")
        circle.inputs[4].default_value = 0.2
        circle.location = (-700, 0)
        set_rad = n.new("GeometryNodeSetCurveRadius")
        set_rad.inputs[2].default_value = 0.05
        set_rad.location = (-450, 0)
        inst = n.new("GeometryNodeInstanceOnPoints")
        inst.inputs[5].default_value = (1.570796, 0, 0)
        inst.location = (-200, 100)
        cap = n.new("GeometryNodeCaptureAttribute")
        cap.domain = "INSTANCE"
        if hasattr(cap, "capture_items"):
            cap.capture_items.new("INT", "Index")
        cap.location = (50, 100)
        idx_in = n.new("GeometryNodeInputIndex")
        idx_in.location = (-200, -100)
        rand = n.new("FunctionNodeRandomValue")
        rand.location = (300, -100)
        ramp = n.new("ShaderNodeValToRGB")
        ramp.location = (550, -100)
        to_gp = n.new("GeometryNodeCurvesToGreasePencil")
        to_gp.location = (300, 100)
        store = n.new("GeometryNodeStoreNamedAttribute")
        store.data_type, store.domain = "FLOAT_COLOR", "POINT"
        store.inputs[2].default_value = "vertex_color"
        store.location = (600, 100)

        links.new(inp.outputs[0], inst.inputs[0])
        links.new(circle.outputs[0], set_rad.inputs[0])
        links.new(set_rad.outputs[0], inst.inputs[2])
        links.new(inst.outputs[0], cap.inputs[0])
        links.new(idx_in.outputs[0], cap.inputs[1])
        links.new(cap.outputs[0], to_gp.inputs[0])
        links.new(to_gp.outputs[0], store.inputs[0])
        links.new(cap.outputs[1], rand.inputs[7])
        links.new(rand.outputs[1], ramp.inputs[0])
        links.new(ramp.outputs[0], store.inputs[3])
        links.new(store.outputs[0], out.inputs[0])
        update_color_ramp(group, 0)

    if "Taichi_GN" not in obj.modifiers:
        mod = obj.modifiers.new("Taichi_GN", "NODES")
        mod.node_group = group
    return obj, group


def sync_mesh_data(scene):
    obj, _ = get_sim_resources()
    n = active_count[None]
    if n == 0:
        return
    mesh = obj.data
    if len(mesh.vertices) != n:
        mesh.clear_geometry()
        mesh.vertices.add(n)
    ti.sync()
    curr_pos = pos.to_numpy()[:n]
    _coords_cache[:n, 0], _coords_cache[:n, 2] = curr_pos[:, 0], curr_pos[:, 1]
    _coords_cache[:n, 1] = 0.0
    mesh.vertices.foreach_set("co", _coords_cache[:n].flatten())
    mesh.update()


@bpy.app.handlers.persistent
def sync_physics_handler(scene):
    if scene.frame_current <= scene.frame_start:
        init_sim()
    is_playing = bpy.context.screen and bpy.context.screen.is_animation_playing
    if not (is_playing or scene.physics_playground_active):
        return
    gravity_val[None], bounce_att[None] = scene.physics_gravity, scene.physics_bounce
    rand_x_vel[None] = int(scene.physics_random_h_vel)
    step_sim(1e-3, 10)
    sync_mesh_data(scene)


# --- 5. INTERACTION & UI ---
def set_viewport_vertex_shading(context):
    for area in context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    try:
                        space.shading.color_type = "VERTEX"
                    except:
                        pass


def draw_boundary_callback(context):
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    b = BOUNDARY
    coords = [(-b, 0, -b), (b, 0, -b), (b, 0, -b), (b, 0, b), (b, 0, b), (-b, 0, b), (-b, 0, b), (-b, 0, -b)]
    batch = batch_for_shader(shader, "LINES", {"pos": coords})
    shader.bind()
    shader.uniform_float("color", (0.1, 0.8, 0.2, 1.0))
    gpu.state.line_width_set(2.0)
    batch.draw(shader)


def update_active_state(self, context):
    handle = bpy.app.driver_namespace.get(HANDLE_KEY)
    if self.physics_playground_active:
        if not handle:
            bpy.app.driver_namespace[HANDLE_KEY] = bpy.types.SpaceView3D.draw_handler_add(draw_boundary_callback, (context,), "WINDOW", "POST_VIEW")
        get_sim_resources()[0].hide_select = True
        set_viewport_vertex_shading(context)
    else:
        if handle:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
            del bpy.app.driver_namespace[HANDLE_KEY]
        obj = bpy.data.objects.get(SIM_NAME)
        if obj:
            obj.hide_select = False


class PHYSICS_OT_playground_modal(bpy.types.Operator):
    bl_idname = "physics.playground_modal"
    bl_label = "Interactive Interaction"
    _last_spawn_time = 0.0  # Track time to debounce clicks

    def modal(self, context, event):
        if context.area:
            context.area.tag_redraw()
        if not context.scene.physics_playground_active:
            return {"FINISHED"}
        if event.type == "ESC":
            context.scene.physics_playground_active = False
            return {"FINISHED"}

        # IMPROVED LOGIC: Handle PRESS only, with a time-based debounce (0.1s)
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            # 1. Check if we just spawned (debounce)
            current_time = time.time()
            if current_time - self.__class__._last_spawn_time < 0.1:
                return {"PASS_THROUGH"}

            # 2. Get viewport info
            region = next((r for r in context.area.regions if r.type == "WINDOW"), None)
            rv3d = context.space_data.region_3d

            # 3. Only spawn if mouse is inside the 3D window area (not over panel)
            if region and rv3d:
                # Check if mouse is actually over the viewport window, not side panels
                if region.x <= event.mouse_x <= region.x + region.width and region.y <= event.mouse_y <= region.y + region.height:
                    coord = (event.mouse_x - region.x, event.mouse_y - region.y)
                    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
                    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

                    if abs(direction.y) > 1e-6:
                        t = -origin.y / direction.y
                        hit = origin + t * direction

                        # Sync and Spawn
                        gravity_val[None], bounce_att[None] = context.scene.physics_gravity, context.scene.physics_bounce
                        rand_x_vel[None] = int(context.scene.physics_random_h_vel)

                        spawn_particle_at(hit.x, hit.z)
                        self.__class__._last_spawn_time = current_time
                        sync_mesh_data(context.scene)

        return {"PASS_THROUGH"}  # ALWAYS pass through so Blender UI and selection work

    def invoke(self, context, event):
        if context.space_data.type == "VIEW_3D":
            context.scene.physics_playground_active = True
            context.window_manager.modal_handler_add(self)
            return {"RUNNING_MODAL"}
        return {"CANCELLED"}


class PHYSICS_PT_playground(bpy.types.Panel):
    bl_label = "Physics Playground"
    bl_idname = "PHYSICS_PT_playground"
    bl_space_type, bl_region_type, bl_category = "VIEW_3D", "UI", "Taichi"

    def draw(self, context):
        layout, s = self.layout, context.scene
        layout.operator("physics.init_playground", icon="FILE_REFRESH")
        if not s.physics_playground_active:
            layout.operator("physics.playground_modal", text="Start Interaction", icon="PLAY")
        else:
            layout.prop(s, "physics_playground_active", text="Stop Interaction", icon="CANCEL", toggle=True)
        col = layout.column(align=True)
        col.prop(s, "physics_gravity")
        col.prop(s, "physics_bounce")
        col.prop(s, "physics_random_h_vel")
        col.prop(s, "physics_swatch_index")
        layout.label(text=f"Particles: {active_count[None]}/{MAX_PARTICLES}")


class PHYSICS_OT_init(bpy.types.Operator):
    bl_idname = "physics.init_playground"
    bl_label = "Reset Simulation"

    def execute(self, context):
        init_sim()
        sync_mesh_data(context.scene)
        return {"FINISHED"}


# --- 6. REGISTRATION ---
classes = (PHYSICS_OT_playground_modal, PHYSICS_OT_init, PHYSICS_PT_playground)


def register():
    bpy.app.handlers.frame_change_pre[:] = [h for h in bpy.app.handlers.frame_change_pre if h.__name__ != "sync_physics_handler"]
    for cls in classes:
        bpy.utils.register_class(cls)
    S = bpy.types.Scene
    S.physics_playground_active = bpy.props.BoolProperty(name="Active", default=False, update=update_active_state)
    S.physics_gravity = bpy.props.FloatProperty(name="Gravity", default=-100.0, min=-1000.0, max=1000.0)
    S.physics_bounce = bpy.props.FloatProperty(name="Bounce", default=0.7, min=0.0, max=1.0)
    S.physics_random_h_vel = bpy.props.BoolProperty(name="Random X", default=True)
    items = [(str(i), f"Palette {i + 1}", "") for i in range(len(SWATCHES))]
    S.physics_swatch_index = bpy.props.EnumProperty(name="Swatch", items=items, default="0", update=lambda s, c: update_color_ramp(get_sim_resources()[1], s.physics_swatch_index))
    bpy.app.handlers.frame_change_pre.append(sync_physics_handler)


def unregister():
    handle = bpy.app.driver_namespace.get(HANDLE_KEY)
    if handle:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        except:
            pass
        del bpy.app.driver_namespace[HANDLE_KEY]
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    S = bpy.types.Scene
    del S.physics_playground_active, S.physics_gravity, S.physics_bounce, S.physics_random_h_vel, S.physics_swatch_index
    bpy.app.handlers.frame_change_pre[:] = [h for h in bpy.app.handlers.frame_change_pre if h.__name__ != "sync_physics_handler"]


if __name__ == "__main__" or __name__ == "<run_path>":
    register()
