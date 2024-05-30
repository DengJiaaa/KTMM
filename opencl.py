import numpy as np
import pyopencl as cl
import time
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import pandas as pd

from constants import au, G, RE, ME, m, r, theta, v


nn = 9
N = 100
dt = 0.1
x = r * np.cos(theta)
y = r * np.sin(theta)
v_x = -v * np.sin(theta)
v_y = v * np.cos(theta)


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

m_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m)
x_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
y_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)
v_x_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v_x)
v_y_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v_y)
accx_0_g = cl.Buffer(ctx, mf.READ_WRITE, size=x.nbytes)
accy_0_g = cl.Buffer(ctx, mf.READ_WRITE, size=y.nbytes)
accx_1_g = cl.Buffer(ctx, mf.READ_WRITE, size=x.nbytes)
accy_1_g = cl.Buffer(ctx, mf.READ_WRITE, size=y.nbytes)

prg = cl.Program(ctx, '''
    int N=100;
    int dt=0.1;
    int nn=9;
    __kernel void openclverlet(
      __global double *m_g,__global double *accx_0_g,__global double *accy_0_g,
      __global double *accx_1_g,__global double *accy_1_g,__global double *x_g,
      __global double *y_g,__global double *v_x_g,__global double *v_y_g)
    {
      int gid = get_global_id(0);      
      #一维空间
      double accx_1 = 0;
      double accy_1 = 0;
      for(int j=0;j<nn;j++){
        if(gid!=j){
          double dist_x = x_g[j] - x_g[gid];
          double dist_y = y_g[j] - y_g[gid];
          double r = sqrt(dist_x * dist_x + dist_y * dist_y);
          double factor = m_g[j] / (r * r * r);
          accx_1 += factor * dist_x;
          accy_1 += factor * dist_y;
        }
      }
      accx_1_g[gid] = accx_1;
      accy_1_g[gid] = accy_1;
    }
''').build()

start = time.time()

xxs = [x.tolist()]
yys = [y.tolist()]

for _ in range(N):
    prg.openclverlet(queue, (nn,), None, m_g, accx_0_g, accy_0_g, accx_1_g, accy_1_g, x_g, y_g, v_x_g, v_y_g)

    res_ax_1 = np.empty_like(x)
    cl.enqueue_copy(queue, res_ax_1, accx_1_g)
    res_ay_1 = np.empty_like(y)
    cl.enqueue_copy(queue, res_ay_1, accy_1_g)

    res_ax_0 = np.empty_like(x)
    cl.enqueue_copy(queue, res_ax_0, accx_0_g)
    res_ay_0 = np.empty_like(y)
    cl.enqueue_copy(queue, res_ay_0, accy_0_g)

    res_v_x_g = np.empty_like(x)
    cl.enqueue_copy(queue, res_v_x_g, v_x_g)
    res_v_y_g = np.empty_like(y)
    cl.enqueue_copy(queue, res_v_y_g, v_y_g)
    res_x_g = np.empty_like(x)
    cl.enqueue_copy(queue, res_x_g, x_g)
    res_y_g = np.empty_like(y)
    cl.enqueue_copy(queue, res_y_g, y_g)

    res_v_x_g += 0.5 * (res_ax_0 + res_ax_1) * dt
    res_v_y_g += 0.5 * (res_ay_0 + res_ay_1) * dt
    res_x_g += res_v_x_g * dt + 0.5 * res_ax_1 * dt ** 2
    res_y_g += res_v_y_g * dt + 0.5 * res_ay_1 * dt ** 2

    xxs.append(res_x_g.tolist())
    yys.append(res_y_g.tolist())

    accx_0_g = accx_1_g
    accy_0_g = accy_1_g

print(time.time() - start)

xxs = np.array(xxs)
yys = np.array(yys)
dataframe = pd.DataFrame(
        {'x1': xxs[:, 0], 'y1': yys[:, 0], 'x2': xxs[:, 1], 'y2': yys[:, 1], 'x3': xxs[:, 2], 'y3': yys[:, 2],
         'x4': xxs[:, 3], 'y4': yys[:, 3], 'x5': xxs[:, 4], 'y5': yys[:, 4], 'x6': xxs[:, 5], 'y6': yys[:, 5],
         'x7': xxs[:, 6], 'y7': yys[:, 6], 'x8': xxs[:, 7], 'y8': yys[:, 7], 'x9': xxs[:, 8], 'y9': yys[:, 8]})
dataframe.to_csv("opencl.csv", index=False, sep=',')

#使用OpenCL：OpenCL允许程序利用GPU或其他并行处理设备来加速大规模计算，非常适合处理多体问题，如行星运动模拟。
