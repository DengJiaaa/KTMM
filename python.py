import time
import numpy as np
import random as random
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

from constants import au, G, RE, ME, m, r, theta, v



def particlesGeneration(nn):

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    v_x = -v * np.sin(theta)
    v_y = v * np.cos(theta)
    return m, x, y, v_x, v_y


def pythonverlet(N, dt, m, x, y, v_x, v_y):
    ts = np.arange(0, N * dt, dt)
    xs, ys = [], []
    accx_0 = np.zeros(len(m))
    accy_0 = np.zeros(len(m))
    accx_1 = np.zeros(len(m))
    accy_1 = np.zeros(len(m))
    x_ij = (x - x.reshape(len(m), 1))
    y_ij = (y - y.reshape(len(m), 1))
    r_ij = np.sqrt(x_ij ** 2 + y_ij ** 2)

    for i in range(len(m)):
        for j in range(len(m)):
            if i != j:
                accx_0[i] += (m[j] * x_ij[i, j] / r_ij[i, j] ** 3)
                accy_0[i] += (m[j] * y_ij[i, j] / r_ij[i, j] ** 3)

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

        v_x += 0.5 * (accx_0 + accx_1) * dt
        v_y += 0.5 * (accy_0 + accy_1) * dt
        x += v_x * dt + 0.5 * accx_1 * dt ** 2
        y += v_y * dt + 0.5 * accy_1 * dt ** 2
        xs.append(x.tolist())
        ys.append(y.tolist())
        accx_0 = accx_1
        accy_0 = accy_1
        accx_1 = np.zeros(len(m))
        accy_1 = np.zeros(len(m))

    return np.array(xs), np.array(ys)


def main(nn):
    N = 100
    dt = 0.1
    m, x, y, v_x, v_y = particlesGeneration(nn)
    xs, ys = pythonverlet(N, dt, m, x, y, v_x, v_y)

    #动画
    # def animate(n):
    #     for i in range(len(m)):
    #         traces[i].set_data(xs[:n, i], ys[:n, i])
    #         pts[i].set_data(xs[n, i], ys[n, i])
    #     # k_text.set_text(textTemplate % (ts[n]/3600/24))
    #     return traces + pts + [k_text]
    #
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(xlim=(-31 * RE, 31 * RE), ylim=(-31 * RE, 31 * RE))
    # ax.grid()
    # traces = [ax.plot([], [], '-', lw=0.5)[0] for _ in range(nn)]
    # pts = [ax.plot([], [], marker='o')[0] for _ in range(nn)]
    # k_text = ax.text(0.05, 0.85, '', transform=ax.transAxes)
    # ani = FuncAnimation(fig, animate, range(N), interval=100, blit=True)
    # plt.show()

    dataframe = pd.DataFrame(
        {'x1': xs[:, 0], 'y1': ys[:, 0], 'x2': xs[:, 1], 'y2': ys[:, 1], 'x3': xs[:, 2], 'y3': ys[:, 2],
         'x4': xs[:, 3], 'y4': ys[:, 3], 'x5': xs[:, 4], 'y5': ys[:, 4], 'x6': xs[:, 5], 'y6': ys[:, 5],
         'x7': xs[:, 6], 'y7': ys[:, 6], 'x8': xs[:, 7], 'y8': ys[:, 7], 'x9': xs[:, 8], 'y9': ys[:, 8]})
    dataframe.to_csv("python.csv", index=False, sep=',')


if __name__ == '__main__':
    start = time.time()
    main(nn=9)
    end = time.time()
    print(end - start)
