import numpy as np
cimport numpy as np
import pandas as pd
cimport pandas as pd
import time

au, G, RE, ME = 1.48e11, 6.67e-11, 1.48e11, 5.965e24

def create(int nn):
    filename = f"data_{nn}.csv"
    df = pd.read_csv(filename)

    m = df['m'].values
    x = df['x'].values
    y = df['y'].values
    v_x = df['v_x'].values
    v_y = df['v_y'].values
    return m, x, y, v_x, v_y


def pythonverlet(int N, double dt, np.ndarray[double, ndim=1] m, np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] v_x, np.ndarray[double, ndim=1] v_y):
    cdef int i, j
    cdef double xij, yij, rij
    cdef np.ndarray[double, ndim=1] ts = np.arange(0, N * dt, dt)
    cdef np.ndarray[double, ndim=2] xs = np.zeros((len(ts), len(m)))
    cdef np.ndarray[double, ndim=2] ys = np.zeros((len(ts), len(m)))
    cdef np.ndarray[double, ndim=1] accx_0 = np.zeros(len(m))
    cdef np.ndarray[double, ndim=1] accy_0 = np.zeros(len(m))
    cdef np.ndarray[double, ndim=1] accx_1 = np.zeros(len(m))
    cdef np.ndarray[double, ndim=1] accy_1 = np.zeros(len(m))
    cdef np.ndarray[double, ndim=2] x_ij, y_ij, r_ij

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
    xs[0] = x
    ys[0] = y

    for n in range(1, len(ts)):
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
        xs[n] = x
        ys[n] = y
        accx_0 = accx_1
        accy_0 = accy_1
        accx_1 = np.zeros(len(m))
        accy_1 = np.zeros(len(m))

    return xs, ys


def main(nn):
    N = 500
    dt = 360000
    m, x, y, v_x, v_y = create(nn)
    xs, ys = pythonverlet(N, dt, m, x, y, v_x, v_y)



if __name__ == '__main__':
    start = time.time()
    main(nn=100)
    end = time.time()
    print(end - start)
