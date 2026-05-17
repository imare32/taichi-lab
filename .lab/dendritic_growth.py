"""
https://pages.nist.gov/fipy/en/latest/generated/examples.phase.anisotropy.html#module-examples.phase.anisotropy
Solve a dendritic solidification problem.
"""

import taichi as ti
import numpy as np

# 启用 GPU 加速
ti.init(arch=ti.gpu)

# 1. 网格参数 (与 FiPy 示例完全一致)
nx = 500
ny = 500
dx = 0.025
dy = 0.025

# 2. 物理参数
DT_heat = 2.25
alpha = 0.015
c_ani = 0.02
N_sym = 6.0 # 分支数量
theta_off = np.pi / 8.0
tau = 3e-4
kappa1 = 0.9
kappa2 = 20.0

# 3. 显式计算的稳定性时间步长
dt = 2e-5
substeps = 25  # 25 * 2e-5 = 5e-4 (等效于 FiPy 的单步时间)

# 4. 数据场分配
phi = ti.field(dtype=ti.f32, shape=(nx, ny))
dT_field = ti.field(dtype=ti.f32, shape=(nx, ny))
new_phi = ti.field(dtype=ti.f32, shape=(nx, ny))
new_dT = ti.field(dtype=ti.f32, shape=(nx, ny))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny))

@ti.func
def get_val(field: ti.template(), i: ti.i32, j: ti.i32):
    # 处理 Neumann 零通量边界条件
    i = ti.max(0, ti.min(nx - 1, i))
    j = ti.max(0, ti.min(ny - 1, j))
    return field[i, j]

@ti.func
def compute_flux_x(i, j):
    # 计算右侧面 (i+0.5, j) 的通量 J_x
    dphi_dx = (get_val(phi, i + 1, j) - get_val(phi, i, j)) / dx
    
    # y方向梯度取相邻两中心的平均值
    dy_i = (get_val(phi, i, j + 1) - get_val(phi, i, j - 1)) / (2.0 * dy)
    dy_ip1 = (get_val(phi, i + 1, j + 1) - get_val(phi, i + 1, j - 1)) / (2.0 * dy)
    dphi_dy = 0.5 * (dy_i + dy_ip1)

    # 各向异性物理量计算
    psi = theta_off + ti.atan2(dphi_dy, dphi_dx)
    Phi = ti.tan(N_sym * psi / 2.0)
    PhiSq = Phi**2

    beta = 0.0
    DbetaDpsi = 0.0
    # 防止 tan 函数趋于无穷大导致的数值爆炸
    if PhiSq > 1e10:
        beta = -1.0
        DbetaDpsi = 0.0
    else:
        beta = (1.0 - PhiSq) / (1.0 + PhiSq)
        DbetaDpsi = -N_sym * 2.0 * Phi / (1.0 + PhiSq)

    Ddia = 1.0 + c_ani * beta
    Doff = c_ani * DbetaDpsi
    M = alpha**2 * (1.0 + c_ani * beta)

    # 张量乘法
    Dxx = M * Ddia
    Dxy = M * (-Doff)

    return Dxx * dphi_dx + Dxy * dphi_dy

@ti.func
def compute_flux_y(i, j):
    # 计算上方侧面 (i, j+0.5) 的通量 J_y
    dphi_dy = (get_val(phi, i, j + 1) - get_val(phi, i, j)) / dy

    # x方向梯度取相邻两中心的平均值
    dx_j = (get_val(phi, i + 1, j) - get_val(phi, i - 1, j)) / (2.0 * dx)
    dx_jp1 = (get_val(phi, i + 1, j + 1) - get_val(phi, i - 1, j + 1)) / (2.0 * dx)
    dphi_dx = 0.5 * (dx_j + dx_jp1)

    psi = theta_off + ti.atan2(dphi_dy, dphi_dx)
    Phi = ti.tan(N_sym * psi / 2.0)
    PhiSq = Phi**2

    beta = 0.0
    DbetaDpsi = 0.0
    if PhiSq > 1e10:
        beta = -1.0
        DbetaDpsi = 0.0
    else:
        beta = (1.0 - PhiSq) / (1.0 + PhiSq)
        DbetaDpsi = -N_sym * 2.0 * Phi / (1.0 + PhiSq)

    Ddia = 1.0 + c_ani * beta
    Doff = c_ani * DbetaDpsi
    M = alpha**2 * (1.0 + c_ani * beta)

    Dyx = M * Doff
    Dyy = M * Ddia

    return Dyx * dphi_dx + Dyy * dphi_dy

@ti.kernel
def evolve():
    for i, j in phi:
        # 1. 求解相场方程 (Phase field)
        flux_x_right = compute_flux_x(i, j)
        flux_x_left  = compute_flux_x(i - 1, j)
        flux_y_top   = compute_flux_y(i, j)
        flux_y_bot   = compute_flux_y(i, j - 1)

        # 计算散度
        div_J = (flux_x_right - flux_x_left) / dx + (flux_y_top - flux_y_bot) / dy

        p = phi[i, j]
        t = dT_field[i, j]

        # 计算源项
        m_val = p - 0.5 - (kappa1 / np.pi) * ti.atan2(kappa2 * t, 1.0)
        source = p * (1.0 - p) * m_val

        dphi_dt = (div_J + source) / tau
        new_phi[i, j] = p + dphi_dt * dt

        # 2. 求解热扩散方程 (Heat equation)
        t_r = get_val(dT_field, i + 1, j)
        t_l = get_val(dT_field, i - 1, j)
        t_t = get_val(dT_field, i, j + 1)
        t_b = get_val(dT_field, i, j - 1)
        lap_T = (t_r + t_l + t_t + t_b - 4.0 * t) / (dx**2)

        # 温度变化受相变潜热 (dphi_dt) 的直接影响
        new_dT[i, j] = t + (DT_heat * lap_T + dphi_dt) * dt

@ti.kernel
def update():
    for i, j in phi:
        phi[i, j] = new_phi[i, j]
        dT_field[i, j] = new_dT[i, j]

@ti.kernel
def render():
    for i, j in phi:
        # 自定义渲染配色方案，模拟参考图片的红/蓝风格
        val = phi[i, j]
        r = 0.7 * (1.0 - val) + 0.1 * val  # 液态偏红
        g = 0.1 * (1.0 - val) + 0.2 * val
        b = 0.1 * (1.0 - val) + 0.7 * val  # 固态偏蓝
        
        # 提取固液界面 (高亮黄/白边缘)
        interface = 4.0 * val * (1.0 - val)
        r += interface * 0.6
        g += interface * 0.6
        b += interface * 0.2
        
        pixels[i, j] = ti.Vector([r, g, b])

@ti.kernel
def initialize():
    radius_sq = (dx * 5.0)**2
    center_x = nx * dx / 2.0
    center_y = ny * dy / 2.0

    for i, j in phi:
        x = i * dx
        y = j * dy
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        if dist_sq < radius_sq:
            phi[i, j] = 1.0
        else:
            phi[i, j] = 0.0
            
        dT_field[i, j] = -0.5

def main():
    initialize()
    gui = ti.GUI("Warren Model Dendrite - Taichi", res=(nx, ny))

    step = 0
    while gui.running:
        # 每帧执行多次计算以加快视觉推进速度
        for _ in range(substeps):
            evolve()
            update()
            
        step += 1
        if step % 2 == 0:
            render()
            gui.set_image(pixels)
            gui.show()

if __name__ == '__main__':
    main()