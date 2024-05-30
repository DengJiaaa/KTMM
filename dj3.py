import numpy as np

import matplotlib.pyplot as plt

plt.figure()
N=[100,200,400]
verlet = [6.2451865673065186+6.387768030166626+6.24855637550354/3,
          24.9861421585083+25.322028398513794+26.28005838394165/3,
          104.22692155838013+104.25569512589456+103.98561523487956/3]
verlet = np.array(verlet)
multiprocessing = [5.778174638748169+5.681171655654907+5.502000570297241/3,
                   14.533876895904541+13.493397235870361+13.72015929222107/3,
                   44.62140130996704+45.25698456128759+44.584611654281616/3]
multiprocessing = np.array(multiprocessing)
cython = [0.38262200355529785+0.34964871406555176+0.35567402839660645/3,
           1.4004971981048584+1.5166466236114502+1.3966970443725586/3,
           6.262856721878052+6.351556301116943+6.33368182182312/3]
cython = np.array(cython)
opencl = [0.6042556762695312+0.6265833377838135+0.6425015926361084/3,
          0.5682759284973145+0.6395535469055176+0.6463119983673096/3,
          0.5763847827911377+0.6557183265686035+0.6284389495849609/3]
opencl = np.array(opencl)
plt.plot(N,verlet,'*--',label="python")
plt.plot(N,multiprocessing,'.--',label="multiprocessing")
plt.plot(N,cython,'o--',label="cython")
plt.plot(N,opencl,'h--',label="opencl")
plt.xlabel('N')
plt.ylabel('t')
plt.legend()
plt.savefig("time.jpg")
plt.show()

plt.figure()
multiprocessing_python = verlet/multiprocessing
cython_python = verlet/cython
opencl_python = verlet/opencl
plt.plot(N,multiprocessing_python,'.--',label="multiprocessing_python")
plt.plot(N,cython_python,'o--',label="cython_python")
plt.plot(N,opencl_python,'h--',label="opencl_python")
plt.xlabel('N')
plt.ylabel('t/t')
plt.legend()
plt.savefig("ускорения.jpg")
plt.show()