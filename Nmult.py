import time
import numpy as np
import pandas as pd
import multiprocessing as mp

au, G, RE, ME = 1.48e11, 6.67e-11, 1.48e11, 5.965e24

def create(nn):
    filename = f"data_{nn}.csv"
    df = pd.read_csv(filename)
    m = df['m'].values
    x = df['x'].values
    y = df['y'].values
    v_x = df['v_x'].values
    v_y = df['v_y'].values
    return m, x, y, v_x, v_y

def compute_acceleration(i, m, positions):
    acc_x, acc_y = 0, 0
    x_i, y_i = positions[i]
    for j in range(len(m)):
        if i != j:
            x_j, y_j = positions[j]
            dx, dy = x_j - x_i, y_j - y_i
            r_ij = np.sqrt(dx ** 2 + dy ** 2)
            acc_x += m[j] * dx / r_ij ** 3
            acc_y += m[j] * dy / r_ij ** 3
    return (acc_x, acc_y)

def update_positions_and_velocities(args):
    i, m, positions, velocities, acc_0, dt = args
    acc_x, acc_y = compute_acceleration(i, m, positions)
    v_x, v_y = velocities[i]
    x, y = positions[i]

    # Update velocities and positions using the Verlet method
    v_x += 0.5 * (acc_0[i][0] + acc_x) * dt
    v_y += 0.5 * (acc_0[i][1] + acc_y) * dt
    x += v_x * dt + 0.5 * acc_x * dt ** 2
    y += v_y * dt + 0.5 * acc_y * dt ** 2

    return i, (x, y), (v_x, v_y), (acc_x, acc_y)
#使用的进程数：进程数通常等于机器的 CPU 核心数，除非手动指定。这意味着在四核机器上默认会有四个进程，每个进程可能分别计算一部分行星的运动（取决于行星的总数和任务划分）。
def pythonverlet(N, dt, m, x, y, v_x, v_y):
    positions = np.column_stack((x, y))
    velocities = np.column_stack((v_x, v_y))
    acc_0 = np.array([compute_acceleration(i, m, positions) for i in range(len(m))])

    pool = mp.Pool(mp.cpu_count())
    for _ in range(N):
        args = [(i, m, positions, velocities, acc_0, dt) for i in range(len(m))]
        results = pool.map(update_positions_and_velocities, args)
        for i, pos, vel, acc in results:
            positions[i], velocities[i], acc_0[i] = pos, vel, acc

    pool.close()
    pool.join()
    return positions.T[0], positions.T[1]

def main(nn):
    N = 500
    dt = 360000
    m, x, y, v_x, v_y = create(nn)
    xs, ys = pythonverlet(N, dt, m, x, y, v_x, v_y)
    print("Simulation completed.")

if __name__ == '__main__':
    start = time.time()
    main(nn=400)
    end = time.time()
    print(f"Elapsed time: {end - start} seconds")

# 工作分配方式：
# 每个进程的任务：
#
# 每个进程负责计算特定行星在给定时间步长内的位置、速度和加速度更新。具体来说，计算每个行星在其他所有行星的引力影响下的加速度、更新其速度和位置。
# 如何分配：
#
# pythonverlet 函数中初始化了所有行星的位置和速度。
# 对于每个时间步，该函数构建一个任务列表，每个任务包含一个行星的当前位置、速度、上一时间步的加速度、质量数组和时间步长等信息。
# 这个任务列表被传递给 Pool.map 方法，该方法将任务分配给进程池中的每个进程。每个进程接收一个任务（即一组计算一个行星的数据），执行 update_positions_and_velocities 函数进行计算。
# update_positions_and_velocities 函数中的计算：
#
# 每个进程首先调用 compute_acceleration 来为其分配的行星计算新的加速度。
# 根据 Verlet 方法，使用当前和上一时刻的加速度来更新行星的速度和位置。
# 返回更新后的位置、速度和新的加速度。