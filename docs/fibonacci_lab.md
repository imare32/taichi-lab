# Fibonacci 实验室 (生成模式)

此模块实现了一个基于 Taichi 加速的 Fibonacci 球体/半球体生成器，支持多种拓扑连接模式，包括叶序 (Phyllotaxis) 螺旋和 Voronoi 对偶转换。

## 1. 架构模式: 多组件生成 (Multi-Component Generation)

该实验室采用 **中心化属性驱动 (Property-Driven)** 的生成架构。一个核心参数集控制多个生成对象：

- **核心对象 (`Fibonacci_Lab`)**: 基础点云。
- **螺旋对象 (`Fibonacci_Spiral`)**: 包含基于叶序率生成的边和面的螺旋结构。
- **Voronoi 对象 (`Fibonacci_Voronoi`)**: 基础点的 3D Voronoi 对偶网格。

## 2. 技术规格

### 分布算法 (Taichi)
使用 `compute_fibonacci_points_kernel` 并行计算球面上均匀分布的点：
- **黄金比例**: 基于 $\frac{1+\sqrt{5}}{2}$ 计算 phi 角。
- **几何模式**: 
    - **全球模式 (SPHERE)**: 映射到 $z \in [-1, 1]$。
    - **半球模式 (HEMISPHERE)**: 映射到 $z \in [0, 1]$。

### 拓扑连接
支持三种连接逻辑：
- **SEQUENTIAL** (顺序): 按生成顺序连接点 (0-1-2...)。
- **PHYLLOTAXIS** (叶序): 模拟植物生长的双螺旋结构。
    - **顺时针/逆时针步长**: 使用斐波那契数（如 13, 21, 34）作为步长偏移量来连接点。
- **VORONOI 对偶 (CPU)**: 
    - 使用 `bmesh.ops.convex_hull` 计算点集的凸包。
    - 通过凸包的对偶转换生成 Voronoi 单元。
    - **半球修剪**: 在半球模式下，使用 `bmesh.ops.bisect_plane` 沿 Z=0 平面精确修剪 Voronoi 单元。

## 3. 视觉与几何节点

建议与几何节点结合使用：
- **实例采样 (Instancing)**: 使用 `Instance on Points` 在生成的点云上分布高精度几何体。
- **着色属性**: 利用 `seed_pos` 属性为每个 Voronoi 单元分配不同的随机颜色或材质参数。

## 4. 数据接口

生成的各组件暴露以下属性：

| 属性 | 域 (Domain) | 类型 | 描述 |
| :--- | :--- | :--- | :--- |
| `position` | 点 (Point) | `FLOAT_VECTOR` | 顶点的 3D 坐标。 |
| `seed_pos` | 面 (Face) | `FLOAT_VECTOR` | 仅限 Voronoi 对象。每个单元对应的原始 Fibonacci 种子位置，用于基于位置的着色逻辑。 |

## 5. 实时同步

- **混合模式更新**:
    - **对象模式 (`sync_mesh_object`)**: 使用 `foreach_set` 快速更新大量顶点的坐标。
    - **编辑模式 (`sync_mesh_edit`)**: 支持外科手术式 BMesh 更新，允许用户在保持仿真运行的同时选择或手动微调生成的顶点/面。
- **属性同步**: `face_attrs` 逻辑确保自定义数据（如 `seed_pos`）在同步过程中保持一致。

## 6. UI/UX 工作流

1. **设置规模**: 调整 **Points** 计数。支持从 2 到 50,000 个点的高性能生成。
2. **选择几何**: 在全球和半球模式之间切换。
3. **连接模式**: 选择 **Phyllotaxis** 以查看标志性的斐波那契螺旋模式。
4. **组件管理**: UI 面板提供一键选择按钮和视口/渲染可见性开关，方便管理多个生成的组件。
5. **参数微调**: 对于叶序模式，调整 **Step CW** 和 **Step CCW** 以探索不同的螺旋密度。
