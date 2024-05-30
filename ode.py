import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import time

from constants import au, G, RE, ME, m, r, theta, v

start = time.time()

x = r * np.cos(theta)
y = r * np.sin(theta)
v_x = -v * np.sin(theta)
v_y = v * np.cos(theta)


# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(xlim=(-31*RE,31*RE),ylim=(-31*RE,31*RE))
# ax.grid()
# traces = [ax.plot([],[],'-', lw=0.5)[0] for _ in range(9)]
# pts = [ax.plot([],[],marker='o')[0] for _ in range(9)]
# k_text = ax.text(0.05,0.85,'',transform=ax.transAxes)
# textTemplate = 't = %.3f days\n'

N = 100
dt = 0.1
ts =  np.arange(0,N*dt,dt)
xs,ys = [],[]

def sys_of_funcs(init,t,x_ij,r_ij,m,i):
    r0,v0=init
    f=0
    for j in range(len(m)):
        if i != j:
            f+=(m[j]*x_ij[i,j]/r_ij[i,j]**3)
    return np.array([v0,f])
#print(x,y)
x_ij = (x-x.reshape(len(m),1))
y_ij = (y-y.reshape(len(m),1))
r_ij = np.sqrt(x_ij**2+y_ij**2)
#print(r_ij)
plt.grid()
xx=np.zeros(len(m))
yy=np.zeros(len(m))
for i in range(len(m)):
    result = odeint(sys_of_funcs, (x[i],v_x[i]), ts, args=(x_ij, r_ij, m, i))
    result2 = odeint(sys_of_funcs, (y[i],v_y[i]), ts, args=(y_ij, r_ij, m, i))
    plt.plot(result[:,0],result2[:,0])
    x[i]=result[-1,0]
    y[i]=result2[-1,0]
    v_x[i]=result[-1,-1]
    v_y[i]=result2[-1,-1]

xs.append(x.tolist())
ys.append(y.tolist())
tts =  np.arange(0,N*dt,dt)
for _ in tts:
    x_ij = (x - x.reshape(len(m), 1))
    y_ij = (y - y.reshape(len(m), 1))
    r_ij = np.sqrt(x_ij ** 2 + y_ij ** 2)
    for i in range(len(m)):
        result = odeint(sys_of_funcs, (x[i], v_x[i]), ts, args=(x_ij, r_ij, m, i))
        result2 = odeint(sys_of_funcs, (y[i], v_y[i]), ts, args=(y_ij, r_ij, m, i))
        #plt.plot(result[:,0],result2[:,0])
        #print(result[:, 0], result2[:, 0])
        x[i]=result[-1,0]
        y[i]=result2[-1,0]
        v_x[i] = result[-1, -1]
        v_y[i] = result2[-1, -1]
    xs.append(x.tolist())
    ys.append(y.tolist())
xs = np.array(xs)
ys = np.array(ys)
#plt.show()
#print(len(xs),len(xs[0]))


# def animate(n):
#     for i in range(len(m)):
#         traces[i].set_data(xs[:n,i],ys[:n,i])
#         pts[i].set_data(xs[n,i],ys[n,i])
#     #k_text.set_text(textTemplate % (ts[n]/3600/24))
#     return traces+pts+[k_text]
end=time.time()
print(end-start)

dataframe = pd.DataFrame({'x1': xs[:,0], 'y1': ys[:,0],'x2': xs[:,1], 'y2': ys[:,1],'x3': xs[:,2], 'y3': ys[:,2],
                              'x4': xs[:,3], 'y4': ys[:,3],'x5': xs[:,4], 'y5': ys[:,4],'x6': xs[:,5], 'y6': ys[:,5],
                              'x7': xs[:,6], 'y7': ys[:,6],'x8': xs[:,7], 'y8': ys[:,7],'x9': xs[:,8], 'y9': ys[:,8]})
dataframe.to_csv("ode.csv", index=False, sep=',')
# #
# N=1000
# ani = FuncAnimation(fig, animate,
#     range(N), interval=100, blit=True)
# plt.show()


