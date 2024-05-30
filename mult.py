import time
import numpy as np
import pandas as pd
import multiprocessing as mp

from constants import au, G, RE, ME, m, r, theta, v



def particlesGeneration(nn):

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    v_x = -v * np.sin(theta)
    v_y = v * np.cos(theta)
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
    N = 100
    dt = 0.1
    m, x, y, v_x, v_y = particlesGeneration(nn)
    xs, ys = pythonverlet(N, dt, m, x, y, v_x, v_y)
    # print("Simulation completed.")
    xs = np.array(xs)
    ys = np.array(ys)
    print(ys)
    dataframe = pd.DataFrame(
        {'x1': [xs[0]], 'y1': [ys[0]], 'x2': [xs[1]], 'y2': [ys[1]], 'x3': [xs[2]], 'y3': [ys[2]],
         'x4': [xs[3]], 'y4': [ys[3]], 'x5': [xs[4]], 'y5': [ys[4]], 'x6': [xs[5]], 'y6': [ys[5]],
         'x7': [xs[6]], 'y7': [ys[6]], 'x8': [xs[7]], 'y8': [ys[7]], 'x9': [xs[8]], 'y9': [ys[8]]})
    dataframe.to_csv("mult.csv", index=False, sep=',')

if __name__ == '__main__':
    start = time.time()
    main(nn=9)
    end = time.time()
    print(f"Elapsed time: {end - start} seconds")




