import numpy as np
import pyopencl as cl
import time
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import pandas as pd

nn = 400
N = 500
dt = 360000
au, G, RE, ME = 1.48e11, 6.67e-11, 1.48e11, 5.965e24

filename = f"data_{nn}.csv"
df = pd.read_csv(filename)

m = df['m'].values
x = df['x'].values
y = df['y'].values
v_x = df['v_x'].values
v_y = df['v_y'].values



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
    int N=500;
    int dt=360000;
    int nn=100;
    __kernel void openclverlet(
      __global double *m_g,__global double *accx_0_g,__global double *accy_0_g,
      __global double *accx_1_g,__global double *accy_1_g,__global double *x_g,
      __global double *y_g,__global double *v_x_g,__global double *v_y_g)
    {
      int gid = get_global_id(0);      
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


# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(xlim=(-31 * RE, 31 * RE), ylim=(-31 * RE, 31 * RE))
# ax.grid()
# traces = [ax.plot([], [], '-', lw=0.5)[0] for _ in range(nn)]
# pts = [ax.plot([], [], marker='o')[0] for _ in range(nn)]
# k_text = ax.text(0.05, 0.85, '', transform=ax.transAxes)
# textTemplate = 't = %.3f days\n'
# def animate(n):
#     for i in range(nn):
#         traces[i].set_data(xxs[:n+1,i], yys[:n+1,i])
#         pts[i].set_data(xxs[n,i], yys[n,i])
#     return traces + pts + [k_text]

# ani = FuncAnimation(fig, animate, range(N), interval=100, blit=True)
# plt.show()
