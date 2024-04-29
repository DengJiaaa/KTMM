import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 定义符号变量
x, y, k1, k_1, k2, k3, k_3 = sp.symbols('x y k1 k_1 k2 k3 k_3')

# 定义方程
dxdt = k1 * (1 - x - y) - k_1 * x - k2 * x * (1 - x - y)**2
dydt = k3 * (1 - x - y)**2 - k_3 * y**2

# 模型常数
k1_val = 0.12
k_1_val = 0.005
k2_val = 1.05
k3_val = 0.0032
k_3_val = 0.003
#k_3_val = [0.0005, 0.001, 0.002, 0.003, 0.004]  # 不同的 k_3 值


# 检查解的情况，并选择一个合适的解用于计算


# 计算雅可比矩阵
jacA = sp.Matrix([dxdt, dydt]).jacobian([x, y])

# 计算雅可比矩阵的迹和行列式
traceA = jacA.trace()
detA = jacA.det()

# 为参数扫描设置数值范围
X = np.arange(0.01, 0.9, 0.001)

# 求解 dydt 对 y 的解
result = sp.solve([dxdt,dydt], y,k1)
# print("解的数量：", len(result))
# for sol in result:
#     print("解：", sol)

# 假设我们选择第一个解继续计算（根据实际情况选择）
y_solution = result[0][0]
k1_solution = result[0][1]

y_solution1 = result[1][0]
k1_solution1 = result[1][1]

print(y_solution)
print(k1_solution)
# print(y_solution1)
# print(k1_solution1)

# 定义 lambda 函数
y_function = sp.lambdify((x, k2, k3, k_1, k_3), y_solution)
k1_function = sp.lambdify((x, k2, k3, k_1, k_3), k1_solution)

# 计算 k1 和 y 的值
K1 = k1_function(X, k2_val, k3_val, k_1_val, k_3_val)
Y = y_function(X, k2_val, k3_val, k_1_val, k_3_val)

# 计算行列式和迹的数值
detA_func = sp.lambdify((x, k1, k2, k3, k_1, k_3), detA.subs(y, y_solution))
traceA_func = sp.lambdify((x, k1, k2, k3, k_1, k_3), traceA.subs(y, y_solution))

# 计算符号解
detA_values = detA_func(X, k1_val, k2_val, k3_val, k_1_val, k_3_val)
traceA_values = traceA_func(X, k1_val, k2_val, k3_val, k_1_val, k_3_val)

# 寻找零点交叉
def find_zeros(values):
    zeros = []
    for i in range(values.shape[0] - 1):
        if values[i] == 0 or values[i] * values[i + 1] < 0:
            zeros.append(i)
    return np.asarray(zeros)


detA_zeros = find_zeros(detA_values)
traceA_zeros = find_zeros(traceA_values)


#绘制轨迹和(x,t)(y,t)

def plot_result1():
    # plt.subplot(1, 2, 1)
    # plt.plot(X, detA_values, label='det(A)')
    # plt.scatter(detA_zeros, np.zeros_like(detA_zeros), color='red', label='saddle-node bifurcation points')
    # plt.title('Determinant of Jacobian')
    # plt.legend()
    # plt.grid(True)

    plt.figure(figsize=(8, 6))
    plt.plot(X, traceA_values)
    plt.title('trace')
    #plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(K1, X, color='pink', label=r'$x(k_{1})$')
    plt.plot(K1, Y, 'g--', label=r'$y(k_{1})$')
    for zero in traceA_zeros:
        plt.plot(K1[zero], X[zero], 'bo', label='hopf')
        plt.plot(K1[zero], Y[zero], 'bo', label='hopf')

    for zero in detA_zeros:
        plt.plot(K1[zero], X[zero], 'y^', label='saddle-node')
        plt.plot(K1[zero], Y[zero], 'y^', label='saddle-node')
    plt.title('в плоскости (x, t) и (y, t)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 运行绘图函数
plot_result1()


# Define the differential equations
def system(z, t, k1, k_1, k2, k3, k_3):
    x, y = z
    dxdt = k1 * (1 - x - y) - k_1 * x - k2 * x * (1 - x - y)**2
    dydt = k3 * (1 - x - y)**2 - k_3 * y**2
    return [dxdt, dydt]

# # 模型常数
# k1_val = 0.12
# k_1_val = 0.005
# k2_val = 1.05
# k3_val = 0.0032
# k_3_val = 0.003
# #k_3_val = [0.0005, 0.001, 0.002, 0.003, 0.004]  # 不同的 k_3 值

# Time vector for integration
t = np.linspace(0.01, 2000, 20000)  # Adjust the max time and steps as needed

# 选择分岔点
detA_zero_index = detA_zeros[0]  # 选择第一个鞍节点分岔
traceA_zero_index = traceA_zeros[0]  # 选择第一个Hopf分岔

# 对应的初始条件
initial_conditions = [
    (X[detA_zero_index], Y[detA_zero_index]),
    (X[traceA_zero_index], Y[traceA_zero_index])
]

# 初始化图形
fig, axs = plt.subplots(3, figsize=(10, 15))

# 对每个分岔点附近的初始条件进行数值求解
for i, (x0, y0) in enumerate(initial_conditions):
    sol = odeint(system, [x0, y0], t, args=(k1_val, k_1_val, k2_val, k3_val, k_3_val))
    x_sol = sol[:, 0]
    y_sol = sol[:, 1]

    # 绘制 (x, t) 和 (y, t)
    axs[0].plot(t, x_sol, label=f'x(t) from bifurcation {i+1}')
    axs[1].plot(t, y_sol, label=f'y(t) from bifurcation {i+1}')

    # 绘制相位平面 (x, y)
    axs[2].plot(x_sol, y_sol, label=f'Trajectory near bifurcation {i+1}')

# 设置图像属性
axs[0].set_xlabel('Time')
axs[0].set_ylabel('x')
axs[0].legend()
axs[0].grid(True)
axs[0].set_title('Dynamics of x over Time')

axs[1].set_xlabel('Time')
axs[1].set_ylabel('y')
axs[1].legend()
axs[1].grid(True)
axs[1].set_title('Dynamics of y over Time')

axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
axs[2].legend()
axs[2].grid(True)
axs[2].set_title('Phase Plane Trajectories')

plt.tight_layout()
plt.show()

################# 双参数分析
#кратности линии 中性线

alk = k_3_val / (k_3_val + k3_val)
x = np.linspace(0.01, 0.9, 1000)
n = len(x)
y = np.zeros(n)
z = np.zeros(n)
k2 = np.zeros(n)
k1 = np.zeros(n)
K2 = np.zeros(n)
K1 = np.zeros(n)
ssp = np.zeros(n)
del_ = np.zeros(n)
Sp = np.zeros(n)
Del = np.zeros(n)

# расчет линии нейтральности
for i in range(n):
    y[i] = (1 - x[i]) * (1 - alk)
    z[i] = (1 - y[i] - x[i])
    k2[i] = (k_1_val * (1 - y[i]) / z[i] + k_3_val + k3_val) / (z[i] * (x[i] - z[i]))
    k1[i] = (k_1_val * x[i] + k2[i] * x[i] * z[i] ** 2) / z[i]
    a11 = - k1[i] - k_1_val - k2[i] * z[i] ** 2 + 2 * k2[i] * x[i] * z[i]
    a12 = - k1[i] + 2 * k2[i] * x[i] * z[i]
    a21 = - k3_val
    a22 = - k3_val - k_3_val
    ssp[i] = a11 + a22
    del_[i] = a11 * a22 - a12 * a21
    if del_[i] <= 0:
        k2[i] = -1

# расчет линии кратности
for i in range(n):
    y[i] = (1 - x[i]) * (1 - alk)
    z[i] = (1 - y[i] - x[i])
    K2[i] = k_1_val * (alk * x[i] + z[i]) / (z[i] ** 2 * (x[i] * alk - z[i]))
    K1[i] = (k_1_val * x[i] + K2[i] * x[i] * (1 - x[i] - y[i]) ** 2) / (1 - x[i] - y[i])
    a11 = - K1[i] - k_1_val - K2[i] * z[i] ** 2 + 2 * K2[i] * x[i] * z[i]
    a12 = - K1[i] + 2 * K2[i] * x[i] * z[i]
    a21 = - k3_val
    a22 = - k3_val - k_3_val
    Sp[i] = a11 + a22
    Del[i] = a11 * a22 - a12 * a21

# построение фазового портрета
plt.figure()
plt.plot(k1[:n], k2[:n], 'tomato', label='линии нейтральности')
plt.plot(K1[:n], K2[:n], 'skyblue', label='линии кратности')
plt.xlabel(r'$k_1$', fontsize=14)
plt.ylabel(r'$k_2$', fontsize=14)
plt.axis([0, 1, 0, 10])
plt.grid(True)
plt.legend()
plt.show()