# Taichi Lab: Development Specifications

This document defines the architectural standards and implementation patterns for the Taichi & Blender integration environment.

---

## 1. Core Lifecycle & Initialization

### Taichi Initialization (Safe Guard)
To support hot-reloading within Blender without crashing the host process, always check for an existing runtime before initializing.
```python
import taichi as ti

if not ti.lang.impl.get_runtime().prog:
    try:
        ti.init(arch=ti.gpu)
    except:
        ti.init(arch=ti.cpu)
```

### Registration & Entry Point
Use a consistent pattern for managing multiple classes. **CRITICAL**: The entry point guard MUST include `"<run_path>"` to support direct execution from VS Code (via the Blender Development extension). Without this, the script will fail to register when triggered from an external IDE.

```python
def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

# Mandatory for both Addon installation and VS Code "Run Script" workflow
if __name__ == "__main__" or __name__ == "<run_path>":
    register()
```

---

## 2. Runtime Handler Management

### App Handlers (Self-Healing Strategy)
For `frame_change_pre` and `depsgraph_update_post`, use **surgical name-based cleanup**. This ensures that even if a previous run crashed, the new execution will "self-heal" by removing any surviving handlers with the same name.
```python

def purge_handlers():
    # Remove existing by name to prevent stacking on re-run
    bpy.app.handlers.frame_change_pre[:] = [
        h for h in bpy.app.handlers.frame_change_pre 
        if h.__name__ != frame_handler.__name__
    ]

```
Call purge_handlers() both in register and unregister methods.

### GPU Draw Handlers (Namespace Persistence)
Unlike app handlers, GPU handlers return an opaque handle that must be tracked to be removed. Use `bpy.app.driver_namespace` to store these handles across script executions.
```python
DRAW_HANDLE_KEY = "PROJECT_NAME_DRAW_HANDLE"

def register():
    # 1. Clean up stale handle from namespace
    handle = bpy.app.driver_namespace.get(DRAW_HANDLE_KEY)
    if handle:
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
    
    # 2. Add new handler and store the handle
    new_handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (), 'WINDOW', 'POST_PIXEL')
    bpy.app.driver_namespace[DRAW_HANDLE_KEY] = new_handle
```

---

## 3. Data Architecture & Sync

### Precision & Casting
- **Internal State**: Precision (e.g., `ti.f32` vs `ti.f64`) should be selected based on the specific simulation requirements. Use `ti.f64` for high-precision physical calculations where accumulation error is a concern.
- **Blender Output (Mandatory)**: Blender operates on single precision. You **must** ensure all data passed to Blender (via `ti.ndarray` or `ti.field` used for mesh synchronization) is cast to `ti.f32`. Failure to do so may lead to data incompatibility or unexpected behavior.

### Fast Coordinate Updates
Use zero-copy ndarrays and `foreach_set` for high-frequency vertex updates in Object Mode.
```python
# Taichi side: use ti.types.ndarray(dtype=ti.f32, ndim=2)
# Blender side:
mesh.vertices.foreach_set("co", pos_ndarray.flatten())
```

### Data Interface Disclosure
Every simulation object must disclose its attributes in the UI panel for Geometry Nodes and Shader compatibility.
```python
layout.label(text="Data Interface", icon='SPREADSHEET')
row = layout.column(align=True).row(align=True)
box = row.box()
box.scale_y = 0.5
r = box.row(align=True)
r.label(text="attr_name")
r.label(text="attr_domain")
```

---

## 4. Mesh Synchronization Strategies

Depending on the target object's mode, use one of the following strategies to sync Taichi data back to Blender.

### sync_mesh_object (Object Mode Fast Path)
Mandatory for high-frequency updates when the target is in Object Mode.
- **Priority**: Maximum performance.
- **Logic**: Use `foreach_set` for coordinate and attribute updates. If the vertex/edge/face count changes, use `clear_geometry()` and `from_pydata()` to rebuild the mesh.
```python
def sync_mesh_object(mesh, verts, edges, faces, attributes=None):
    if len(mesh.vertices) != len(verts):
        mesh.clear_geometry()
        mesh.from_pydata(verts, edges, faces)
    else:
        mesh.vertices.foreach_set("co", verts.flatten())
    # Update attributes via foreach_set on mesh.attributes
    mesh.update()
```

### sync_mesh_edit (Edit Mode Surgical Path)
Used when the user is actively editing the target object and requires real-time simulation feedback.
- **Priority**: Real-time interactivity and viewport stability.
- **Logic**: Use the BMesh API via `bmesh.from_edit_mesh(obj.data)`. Perform surgical updates to vertex coordinates. If counts mismatch, perform a full `bm.clear()` and rebuild.
```python
def sync_mesh_edit(mesh_data, verts, edges, faces):
    bm = bmesh.from_edit_mesh(mesh_data)
    if len(bm.verts) != len(verts):
        bm.clear()
        # Rebuild topology...
    else:
        for i, v in enumerate(bm.verts):
            v.co = verts[i]
    bmesh.update_edit_mesh(mesh_data)
```

---

## 5. Object Reference Pattern & Naming

When a script involves a source object (providing input data) and a target object (receiving simulation results), use the following naming and handling conventions.

### Standard Naming
- **`obj_ref`**: The Reference/Source object.
- **`obj_gen`**: The Generated/Target object.

### Mode-Aware Data Retrieval (`obj_ref`)
The primary goal is to maintain **real-time interaction**, especially when the user is modifying the reference object in Edit Mode.

- **Edit Mode Retrieval**: Use `bmesh.from_edit_mesh(obj_ref.data)` to get live, unsaved vertex positions.
- **Object Mode Retrieval**: Use `obj_ref.data` or `foreach_get` for performance.

### Raw vs. Evaluated Geometry
Choose the geometry source based on the simulation's requirements:
- **Raw Geometry**: Use `obj_ref.data` (Object Mode) or `bmesh.from_edit_mesh` (Edit Mode). Best for using the object as a control cage.
- **Evaluated Geometry**: Use `obj_ref.evaluated_get(depsgraph)` to get the mesh **after** modifiers, shape keys, and physics. Use this when the simulation depends on the final deformed shape.

---

## 6. Interaction & Mode Safety

### Source vs. Target Context
- **Source (Data Provider)**: Should be transparent to mode. Use `bmesh.from_edit_mesh(obj.data)` if the source is in Edit Mode to get live vertex data.
- **Target (Data Receiver)**: 
    - **Object Mode**: Preferred for high-frequency updates and structural changes.
    - **Edit Mode**: Use BMesh API for surgical updates. Gating is mandatory if structural changes (vertex count) are attempted.

### Modal Operator Gating
Interaction logic must be gated by the simulation domain. Return `{'PASS_THROUGH'}` when events occur outside the domain to keep the viewport usable.

---

## 7. Pipeline Integration

### Geometry Nodes Setup
Programmatically generate or update GN setups.
- **Deduplication**: Always check if node groups or modifiers already exist before creating new ones.
- **Attribute Access**: Use `Named Attribute` nodes in GN to read simulation data (e.g., `velocity`, `age`).
- **Normalization**: Normalize simulation values to `[0.0, 1.0]` in Taichi to make them "Color Ramp ready" in Blender.
