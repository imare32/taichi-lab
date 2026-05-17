import bpy


# ==========================================
# 1. 定义操作器 (Modal Operator)
# ==========================================
class ModalMouseTracker(bpy.types.Operator):
    """实时追踪鼠标按下时的拖拽坐标"""

    bl_idname = "view3d.modal_mouse_tracker"
    bl_label = "Start Mouse Tracker"

    def modal(self, context, event):
        scene = context.scene

        # 1. 监听鼠标左键的状态变化
        if event.type == "LEFTMOUSE":
            if event.value == "PRESS":
                scene.mouse_is_dragging = True
                scene.mouse_track_x = event.mouse_region_x
                scene.mouse_track_y = event.mouse_region_y
                # 状态改变时刷新一次 UI
                context.area.tag_redraw()
            elif event.value == "RELEASE":
                scene.mouse_is_dragging = False
                context.area.tag_redraw()

        # 2. 监听鼠标移动事件
        elif event.type == "MOUSEMOVE":
            if scene.mouse_is_dragging:
                # 将坐标写入全局 Scene 属性
                scene.mouse_track_x = event.mouse_region_x
                scene.mouse_track_y = event.mouse_region_y

                # 【核心】强制重绘当前区域的 UI，实现实时显示
                context.area.tag_redraw()

        # 3. 按 ESC 或 右键 退出
        elif event.type in {"RIGHTMOUSE", "ESC"}:
            scene.mouse_is_tracking = False
            scene.mouse_is_dragging = False
            context.area.tag_redraw()
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        context.scene.mouse_is_tracking = True
        # 初始化坐标为 0
        context.scene.mouse_track_x = 0
        context.scene.mouse_track_y = 0

        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}


# ==========================================
# 2. 定义 UI 面板 (Panel)
# ==========================================
class VIEW3D_PT_MouseTracker(bpy.types.Panel):
    """在 3D 视图侧边栏创建面板"""

    bl_space_type = "VIEW_3D"  # 位于 3D 视图
    bl_region_type = "UI"  # 位于 UI 区 (N键侧边栏)
    bl_category = "Mouse Tracker"  # 侧边栏的标签名
    bl_label = "实时鼠标数据面板"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # 如果没有在运行追踪器，显示“开始”按钮
        if not scene.mouse_is_tracking:
            layout.operator(ModalMouseTracker.bl_idname, text="开启实时追踪", icon="PLAY")
        else:
            # 如果正在运行，显示状态和实时数据
            layout.label(text="追踪运行中... (按 ESC 退出)", icon="REC")

            box = layout.box()

            # 显示拖拽状态
            if scene.mouse_is_dragging:
                box.label(text="状态: 🟢 拖拽中")
            else:
                box.label(text="状态: 🔵 等待点击")

            # 显示坐标数据
            row = box.row()
            row.label(text=f"X 坐标: {scene.mouse_track_x}")
            row.label(text=f"Y 坐标: {scene.mouse_track_y}")


# ==========================================
# 3. 注册与注销模块
# ==========================================
classes = (
    ModalMouseTracker,
    VIEW3D_PT_MouseTracker,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # 注册全局变量，用于 Operator 和 Panel 之间的数据传递
    bpy.types.Scene.mouse_track_x = bpy.props.IntProperty(name="Mouse X", default=0)
    bpy.types.Scene.mouse_track_y = bpy.props.IntProperty(name="Mouse Y", default=0)
    bpy.types.Scene.mouse_is_dragging = bpy.props.BoolProperty(name="Is Dragging", default=False)
    bpy.types.Scene.mouse_is_tracking = bpy.props.BoolProperty(name="Is Tracking Operator Active", default=False)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    # 清理全局变量
    del bpy.types.Scene.mouse_track_x
    del bpy.types.Scene.mouse_track_y
    del bpy.types.Scene.mouse_is_dragging
    del bpy.types.Scene.mouse_is_tracking


if __name__ == "__main__" or __name__ == "<run_path>":
    register()

