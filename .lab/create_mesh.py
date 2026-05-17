import bpy
import numpy as np


def create_cube():
    # 生成立方体的8个顶点坐标（直接生成N×3格式，无需转置）
    coords = np.array([[i, j, k] for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]])

    # 立方体的面（每个面由4个顶点组成）
    faces = [
        [0, 1, 3, 2],  # 前面
        [4, 5, 7, 6],  # 后面
        [0, 1, 5, 4],  # 右面
        [2, 3, 7, 6],  # 左面
        [0, 2, 6, 4],  # 下面
        [1, 3, 7, 5],  # 上面
    ]

    # 创建网格数据
    mesh = bpy.data.meshes.new(name="Cube")
    mesh.from_pydata(coords.tolist(), [], faces)
    mesh.update()

    # 创建物体并关联网格
    obj = bpy.data.objects.new("Cube", mesh)

    # 将物体添加到场景中
    bpy.context.collection.objects.link(obj)


def create_torus():
    r = 0.3  # 圆环管的半径
    R = 1  # 圆环中心到管中心的距离
    U = np.linspace(0, 2 * np.pi, 50)  # 圆周方向
    V = np.linspace(0, 2 * np.pi, 30)  # 管截面方向

    # 生成圆环顶点坐标（直接生成N×3格式，无需转置）
    torus = np.array(
        [[np.cos(u) * (R + r * np.cos(v)), np.sin(u) * (R + r * np.cos(v)), r * np.sin(v)] for u in U for v in V]
    )

    # 生成圆环的面（四边形）
    faces = []
    num_u = len(U)
    num_v = len(V)

    for i in range(num_u - 1):
        for j in range(num_v - 1):
            # 每个四边形由4个顶点组成
            v1 = i * num_v + j
            v2 = (i + 1) * num_v + j
            v3 = (i + 1) * num_v + (j + 1)
            v4 = i * num_v + (j + 1)
            faces.append([v1, v2, v3, v4])

    # 创建网格数据
    mesh = bpy.data.meshes.new(name="Torus")
    mesh.from_pydata(torus.tolist(), [], faces)
    mesh.update()

    # 创建物体并关联网格
    obj = bpy.data.objects.new("Torus", mesh)

    # 将物体添加到场景中
    bpy.context.collection.objects.link(obj)


if __name__ == "__main__" or __name__ == "<run_path>":
    create_torus()
