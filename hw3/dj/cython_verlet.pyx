import time
import numpy as np
cimport numpy as np
import pandas as pd
import cython
from libc.math cimport sqrt, cos, sin


cdef extern from "math.h":
    double M_PI



def particlesGeneration(int nn):
    cdef:
        long double au= 1.48e11
        long double G=6.67e-11
        long double RE=1.48e11
        long double ME=5.965e24
        np.ndarray[long double, ndim=1] m= np.array([3.32e5, 0.055, 0.815, 1.0,
                  0.107, 317.8, 95.16, 14.54, 17.14]) * ME * 6.67e-11
        np.ndarray[long double, ndim=1] r = np.array([0.0, 0.387, 0.723, 1.0, 1.524, 5.203,
                  9.537, 19.19, 30.7]) * RE
        np.ndarray[long double, ndim=1] theta = np.array([0.90579977, 4.76568695, 1.34869972, 6.02969388, 2.24714959, 3.45095948,
             3.41281759, 4.32174632, 2.33019222])
        np.ndarray[long double, ndim=1] x =  r * np.cos(theta)
        np.ndarray[long double, ndim=1] y =  r * np.sin(theta)
        np.ndarray[long double, ndim=1] v = np.array([0, 47.89, 35.03, 29.79, 24.13, 13.06, 9.64, 6.81, 5.43]) * 1000
        np.ndarray[long double, ndim=1] v_x =  -v * np.sin(theta)
        np.ndarray[long double, ndim=1] v_y =  v * np.cos(theta)
    return m, x, y, v_x, v_y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.wraparound(False)
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


def main(int nn):
    cdef int N = 100
    cdef double dt = 0.1
    cdef np.ndarray[double, ndim=1] m, x, y, v_x, v_y
    m, x, y, v_x, v_y = particlesGeneration(nn)
    cdef np.ndarray[double, ndim=2] xs, ys
    xs, ys = pythonverlet(N, dt, m, x, y, v_x, v_y)

    dataframe = pd.DataFrame(
        {'x1': xs[:, 0], 'y1': ys[:, 0], 'x2': xs[:, 1], 'y2': ys[:, 1], 'x3': xs[:, 2], 'y3': ys[:, 2],
         'x4': xs[:, 3], 'y4': ys[:, 3], 'x5': xs[:, 4], 'y5': ys[:, 4], 'x6': xs[:, 5], 'y6': ys[:, 5],
         'x7': xs[:, 6], 'y7': ys[:, 6], 'x8': xs[:, 7], 'y8': ys[:, 7], 'x9': xs[:, 8], 'y9': ys[:, 8]})
    dataframe.to_csv("cython.csv", index=False, sep=',')

if __name__ == '__main__':
    start = time.time()
    main(nn=9)
    end = time.time()
    print(end - start)
