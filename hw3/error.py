import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt


data1 = pd.read_csv('ode.csv')
data2 = pd.read_csv('python.csv')
data3 = pd.read_csv('mult.csv')
data4 = pd.read_csv('cython.csv')
data5 = pd.read_csv('opencl.csv')
#print(data1[['x1']].iloc[:,0],data3[['x1']].iloc[:,0])


x1=np.sqrt((data1[['x1']].iloc[-1,0]-data2[['x1']].iloc[-1,0])**2 + (data1[['y1']].iloc[-1,0]-data2[['y1']].iloc[-1,0])**2)/pow(10,12)
x2=np.sqrt((data1[['x2']].iloc[-1,0]-data2[['x2']].iloc[-1,0])**2 + (data1[['y2']].iloc[-1,0]-data2[['y2']].iloc[-1,0])**2)/pow(10,12)
x3=np.sqrt((data1[['x3']].iloc[-1,0]-data2[['x3']].iloc[-1,0])**2 + (data1[['y3']].iloc[-1,0]-data2[['y3']].iloc[-1,0])**2)/pow(10,12)
x4=np.sqrt((data1[['x4']].iloc[-1,0]-data2[['x4']].iloc[-1,0])**2 + (data1[['y4']].iloc[-1,0]-data2[['y4']].iloc[-1,0])**2)/pow(10,12)
x5=np.sqrt((data1[['x5']].iloc[-1,0]-data2[['x5']].iloc[-1,0])**2 + (data1[['y5']].iloc[-1,0]-data2[['y5']].iloc[-1,0])**2)/pow(10,12)
x6=np.sqrt((data1[['x6']].iloc[-1,0]-data2[['x6']].iloc[-1,0])**2 + (data1[['y6']].iloc[-1,0]-data2[['y6']].iloc[-1,0])**2)/pow(10,12)
x7=np.sqrt((data1[['x7']].iloc[-1,0]-data2[['x7']].iloc[-1,0])**2 + (data1[['y7']].iloc[-1,0]-data2[['y7']].iloc[-1,0])**2)/pow(10,12)
x8=np.sqrt((data1[['x8']].iloc[-1,0]-data2[['x8']].iloc[-1,0])**2 + (data1[['y8']].iloc[-1,0]-data2[['y8']].iloc[-1,0])**2)/pow(10,12)
x9=np.sqrt((data1[['x9']].iloc[-1,0]-data2[['x9']].iloc[-1,0])**2 + (data1[['y9']].iloc[-1,0]-data2[['y9']].iloc[-1,0])**2)/pow(10,12)

xx=[x1,x2,x3,x4,x5,x6,x7,x8,x9]
print(xx)
xx=array(xx)
xx=xx.flatten()

x1=np.sqrt((data1[['x1']].iloc[-1,0]-data3['x1'].iloc[0])**2 + (data1[['y1']].iloc[-1,0]-data3['y1'].iloc[0])**2)/pow(10,12)
x2=np.sqrt((data1[['x2']].iloc[-1,0]-data3['x2'].iloc[0])**2 + (data1[['y2']].iloc[-1,0]-data3['y2'].iloc[0])**2)/pow(10,12)
x3=np.sqrt((data1[['x3']].iloc[-1,0]-data3['x3'].iloc[0])**2 + (data1[['y3']].iloc[-1,0]-data3['y3'].iloc[0])**2)/pow(10,12)
x4=np.sqrt((data1[['x4']].iloc[-1,0]-data3['x4'].iloc[0])**2 + (data1[['y4']].iloc[-1,0]-data3['y4'].iloc[0])**2)/pow(10,12)
x5=np.sqrt((data1[['x5']].iloc[-1,0]-data3['x5'].iloc[0])**2 + (data1[['y5']].iloc[-1,0]-data3['y5'].iloc[0])**2)/pow(10,12)
x6=np.sqrt((data1[['x6']].iloc[-1,0]-data3['x6'].iloc[0])**2 + (data1[['y6']].iloc[-1,0]-data3['y6'].iloc[0])**2)/pow(10,12)
x7=np.sqrt((data1[['x7']].iloc[-1,0]-data3['x7'].iloc[0])**2 + (data1[['y7']].iloc[-1,0]-data3['y7'].iloc[0])**2)/pow(10,12)
x8=np.sqrt((data1[['x8']].iloc[-1,0]-data3['x8'].iloc[0])**2 + (data1[['y8']].iloc[-1,0]-data3['y8'].iloc[0])**2)/pow(10,12)
x9=np.sqrt((data1[['x9']].iloc[-1,0]-data3['x9'].iloc[0])**2 + (data1[['y9']].iloc[-1,0]-data3['y9'].iloc[0])**2)/pow(10,12)


xx2=[x1,x2,x3,x4,x5,x6,x7,x8,x9]
print(xx2)
xx2=array(xx2)
xx2=xx2.flatten()

x1=np.sqrt((data1[['x1']].iloc[-1,0]-data4[['x1']].iloc[-1,0])**2 + (data1[['y1']].iloc[-1,0]-data4[['y1']].iloc[-1,0])**2)/pow(10,12)
x2=np.sqrt((data1[['x2']].iloc[-1,0]-data4[['x2']].iloc[-1,0])**2 + (data1[['y2']].iloc[-1,0]-data4[['y2']].iloc[-1,0])**2)/pow(10,12)
x3=np.sqrt((data1[['x3']].iloc[-1,0]-data4[['x3']].iloc[-1,0])**2 + (data1[['y3']].iloc[-1,0]-data4[['y3']].iloc[-1,0])**2)/pow(10,12)
x4=np.sqrt((data1[['x4']].iloc[-1,0]-data4[['x4']].iloc[-1,0])**2 + (data1[['y4']].iloc[-1,0]-data4[['y4']].iloc[-1,0])**2)/pow(10,12)
x5=np.sqrt((data1[['x5']].iloc[-1,0]-data4[['x5']].iloc[-1,0])**2 + (data1[['y5']].iloc[-1,0]-data4[['y5']].iloc[-1,0])**2)/pow(10,12)
x6=np.sqrt((data1[['x6']].iloc[-1,0]-data4[['x6']].iloc[-1,0])**2 + (data1[['y6']].iloc[-1,0]-data4[['y6']].iloc[-1,0])**2)/pow(10,12)
x7=np.sqrt((data1[['x7']].iloc[-1,0]-data4[['x7']].iloc[-1,0])**2 + (data1[['y7']].iloc[-1,0]-data4[['y7']].iloc[-1,0])**2)/pow(10,12)
x8=np.sqrt((data1[['x8']].iloc[-1,0]-data4[['x8']].iloc[-1,0])**2 + (data1[['y8']].iloc[-1,0]-data4[['y8']].iloc[-1,0])**2)/pow(10,12)
x9=np.sqrt((data1[['x9']].iloc[-1,0]-data4[['x9']].iloc[-1,0])**2 + (data1[['y9']].iloc[-1,0]-data4[['y9']].iloc[-1,0])**2)/pow(10,12)

xx3=[x1,x2,x3,x4,x5,x6,x7,x8,x9]

xx3=array(xx3)
xx3=xx3.flatten()

x1=np.sqrt((data1[['x1']].iloc[-1,0]-data5[['x1']].iloc[-1,0])**2 + (data1[['y1']].iloc[-1,0]-data5[['y1']].iloc[-1,0])**2)/pow(10,12)
x2=np.sqrt((data1[['x2']].iloc[-1,0]-data5[['x2']].iloc[-1,0])**2 + (data1[['y2']].iloc[-1,0]-data5[['y2']].iloc[-1,0])**2)/pow(10,12)
x3=np.sqrt((data1[['x3']].iloc[-1,0]-data5[['x3']].iloc[-1,0])**2 + (data1[['y3']].iloc[-1,0]-data5[['y3']].iloc[-1,0])**2)/pow(10,12)
x4=np.sqrt((data1[['x4']].iloc[-1,0]-data5[['x4']].iloc[-1,0])**2 + (data1[['y4']].iloc[-1,0]-data5[['y4']].iloc[-1,0])**2)/pow(10,12)
x5=np.sqrt((data1[['x5']].iloc[-1,0]-data5[['x5']].iloc[-1,0])**2 + (data1[['y5']].iloc[-1,0]-data5[['y5']].iloc[-1,0])**2)/pow(10,12)
x6=np.sqrt((data1[['x6']].iloc[-1,0]-data5[['x6']].iloc[-1,0])**2 + (data1[['y6']].iloc[-1,0]-data5[['y6']].iloc[-1,0])**2)/pow(10,12)
x7=np.sqrt((data1[['x7']].iloc[-1,0]-data5[['x7']].iloc[-1,0])**2 + (data1[['y7']].iloc[-1,0]-data5[['y7']].iloc[-1,0])**2)/pow(10,12)
x8=np.sqrt((data1[['x8']].iloc[-1,0]-data5[['x8']].iloc[-1,0])**2 + (data1[['y8']].iloc[-1,0]-data5[['y8']].iloc[-1,0])**2)/pow(10,12)
x9=np.sqrt((data1[['x9']].iloc[-1,0]-data5[['x9']].iloc[-1,0])**2 + (data1[['y9']].iloc[-1,0]-data5[['y9']].iloc[-1,0])**2)/pow(10,12)



xx4=[x1,x2,x3,x4,x5,x6,x7,x8,x9]
xx4=array(xx4)
xx4=xx4.flatten()
# print(xx2)

l1_norm = np.linalg.norm(xx)
# print("xxL2 范数:", l1_norm)
l2_norm = np.linalg.norm(xx2)
# print("xx2L2 范数:", l2_norm)
l3_norm = np.linalg.norm(xx3)
# # print("xx3L2 范数:", l3_norm)
l4_norm = np.linalg.norm(xx4)
# print("xx4L2 范数:", l4_norm)


plt.figure()
plt.title("Погрешность")
nn=np.linspace(1, 4, 4)
N = [l1_norm, l2_norm, l3_norm, l4_norm]
#N = [l1_norm, l2_norm, l4_norm]
print(N)
plt.plot(nn,N,'*')
plt.plot(nn,N,'--')
plt.savefig("error.jpg")
plt.show()