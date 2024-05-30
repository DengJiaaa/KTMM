import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
import time
import pandas as pd
from constants import au, G, RE, ME, m, r, theta, v


def create (nn):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    v_x = -v * np.sin(theta)
    v_y = v * np.cos(theta)
    return m, x, y, v_x, v_y

#模拟 N 个时间步长，每步长为 dt
def pythonverlet(N,dt,m,x,y,v_x,v_y):
    ts = np.arange(0, N * dt, dt)
    xs, ys = [], []
    accx_0 = np.zeros(len(m))
    accy_0 = np.zeros(len(m))
    accx_1 = np.zeros(len(m))
    accy_1 = np.zeros(len(m))
    x_ij = (x - x.reshape(len(m), 1))
    y_ij = (y - y.reshape(len(m), 1))
    # print(x_ij[0])
    r_ij = np.sqrt(x_ij ** 2 + y_ij ** 2)
    for i in range(len(m)):
        for j in range(len(m)):
            if i != j:
                accx_0[i] += (m[j] * x_ij[i, j] / r_ij[i, j] ** 3)
                accy_0[i] += (m[j] * y_ij[i, j] / r_ij[i, j] ** 3)
        # print(accx_0[i], accy_0[i])
    x += v_x * dt + 0.5 * accx_0 * dt ** 2
    y += v_y * dt + 0.5 * accy_0 * dt ** 2
    xs.append(x.tolist())
    ys.append(y.tolist())

    for _ in ts:
        x_ij = (x - x.reshape(len(m), 1))
        y_ij = (y - y.reshape(len(m), 1))
        r_ij = np.sqrt(x_ij ** 2 + y_ij ** 2)
        for i in range(len(m)):
            for j in range(len(m)):
                if i != j:
                    accx_1[i] += (m[j] * x_ij[i, j] / r_ij[i, j] ** 3)
                    accy_1[i] += (m[j] * y_ij[i, j] / r_ij[i, j] ** 3)
                    # print(accx_1[i], accy_1[i])
        v_x += 0.5 * (accx_0 + accx_1) * dt
        v_y += 0.5 * (accy_0 + accy_1) * dt
        x += v_x * dt + 0.5 * accx_1 * dt ** 2
        y += v_y * dt + 0.5 * accy_1 * dt ** 2
        # print(x,y)
        xs.append(x.tolist())
        ys.append(y.tolist())
        accx_0 = accx_1
        accy_0 = accy_1
        accx_1 = np.zeros(len(m))
        accy_1 = np.zeros(len(m))
    xs = np.array(xs)
    ys = np.array(ys)

    return xs,ys

def main():
    N = 500
    dt = 360000
    m, x, y, v_x, v_y = create(9)
    xs, ys = pythonverlet(N, dt, m, x, y, v_x, v_y)

    def animate(n):
        for i in range(len(m)):
            traces[i].set_data(xs[:n, i], ys[:n, i])
            pts[i].set_data(xs[n, i], ys[n, i])
        return traces + pts

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(xlim=(-31 * RE, 31 * RE), ylim=(-31 * RE, 31 * RE))
    ax.grid()
    traces = [ax.plot([], [], '-', lw=0.5)[0] for _ in range(9)]
    pts = [ax.plot([], [], marker='o')[0] for _ in range(9)]
    #创建动画
    ani = FuncAnimation(fig, animate,
                        range(N), interval=100, blit=True)
    plt.show()
    ani.save("pysolar.gif", writer='pillow')



if __name__ == '__main__':
    main()