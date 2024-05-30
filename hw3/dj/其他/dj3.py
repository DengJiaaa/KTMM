import numpy as np

import matplotlib.pyplot as plt

plt.figure()
N=[100,200,400]
verlet = [3.87+3.08+2.94/3, 12.15+11.57+12.08/3, 46.12+47.39+52.38/3]
verlet = np.array(verlet)
multiprocessing = [5.08+3.19+4.08/3,12.32+10.65+11.41/3,42.25+44.68+45.93/3]
multiprocessing = np.array(multiprocessing)
cython = [1.44+1.38+1.46/3,5.74+5.86+6.00/3,23.26+26.14+23.21/3]
cython = np.array(cython)
opencl = [1.08+1.24+0.81/3,2.63+2.71+2.80/3,3.49+3.68+3.72/3]
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