import sys
import numpy as np
import pandas as pd
from PyQt6.QtGui import QAction
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from PyQt6.QtWidgets import QApplication, QMainWindow

def sys_of_funcs(T, t, ci, kij, epsi, S):
    c0 = 5.67
    y1 = T[0]
    y2 = T[1]
    y3 = T[2]
    y4 = T[3]
    y5 = T[4]
    f1 = (1 / ci[0]) * (-kij[0] * (y2 - y1) - epsi[0] * S[0] * c0 * ((y1 / 100) ** 4))
    f2 = (1 / ci[1]) * (
            -kij[0] * (y2 - y1) - kij[1] * (y3 - y2) - epsi[1] * S[1] * c0 * ((y2 / 100) ** 4) + (20 + (-3) * np.sin(t * 1/4) * 100))
    f3 = (1 / ci[2]) * (-kij[1] * (y3 - y2) - kij[2] * (y4 - y3) - epsi[2] * S[2] * c0 * ((y3 / 100) ** 4))
    f4 = (1 / ci[3]) * (-kij[2] * (y4 - y3) - kij[3] * (y5 - y4) - epsi[3] * S[3] * c0 * ((y4 / 100) ** 4))
    f5 = (1 / ci[4]) * (-kij[3] * (y5 - y4) - epsi[4] * S[4] * c0 * ((y5 / 100) ** 4))
    return [f1, f2, f3, f4, f5]
#求解方程的函数


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 200, 200)

        self.initialization_button1 = QAction("S_i и S_ij", self)
        self.initialization_button1.triggered.connect(self.initialization_params1)
        self.initialization_button2 = QAction("график", self)
        self.initialization_button2.triggered.connect(self.initialization_params2)

        toolbar1 = self.addToolBar("Tools")
        toolbar2 = self.addToolBar("Tools")
        toolbar1.addAction(self.initialization_button1)
        toolbar2.addAction(self.initialization_button2)

    def initialization_params2(self):
        mesh = Mesh("model1.obj")
        mesh.plot()

    def initialization_params1(self):
        mesh = Mesh("model1.obj")
        print("Si")
        print(mesh.si)
        print("//")
        print("Sij")
        print(mesh.sij)


class Mesh:
    def __init__(self, file):
        self.vertices, self.faces, self.ver = self.loadmesh(file)
        self.si, self.sij = self.count()

    def loadmesh(self, file):
        vertices = [[] for i in range(5)]
        faces = [[] for i in range(5)]
        ver = []
        j = 0
        with open(file, 'r') as file:
            for line in file:
                strs = line.split(" ")
                if strs[0] == "g":
                    j = j + 1
                elif strs[0] == "f":
                    faces[j - 1].append((int(strs[1]), int(strs[2]), int(strs[3])))
                elif strs[0] == "v":
                    vertices[j].append((float(strs[2]), float(strs[3]), float(strs[4])))
                    ver.append((float(strs[2]), float(strs[3]), float(strs[4])))
        return vertices, faces, ver



    def distance(self, p1, p2):
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)

    def area(self, point1, point2, point3):
        a = self.distance(point1, point2)
        b = self.distance(point2, point3)
        c = self.distance(point3, point1)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        return area

    def count(self):
        si = np.zeros(5)
        sij = np.zeros((5, 5))
        for i in range(5):
            z = []
            for element in (self.vertices[i]):
                max = 0
                min = 0
                z.append(element[1])
            for element in (self.faces[i]):
                if np.min([self.ver[element[0] - 1][1], self.ver[element[1] - 1][1],
                           self.ver[element[2] - 1][1]]) == np.max(z):
                    max = max + self.area(self.ver[element[0] - 1], self.ver[element[1] - 1], self.ver[element[2] - 1])
                elif np.max([self.ver[element[0] - 1][1], self.ver[element[1] - 1][1],
                             self.ver[element[2] - 1][1]]) == np.min(z):
                    min = min + self.area(self.ver[element[0] - 1], self.ver[element[1] - 1], self.ver[element[2] - 1])
            #print(min, max, i, np.min(z), np.max(z))
            if i == 0:
                sij[i][i + 1] = sij[i + 1][i] = max
            elif i != 4:
                sij[i][i + 1] = sij[i + 1][i] = max
                sij[i - 1][i] = sij[i][i - 1] = min
        for i in range(5):
            for element in (self.faces[i]):
                si[i] = si[i] + self.area(self.ver[element[0] - 1], self.ver[element[1] - 1],
                                          self.ver[element[2] - 1])
            # print(S_i[i])
            if i == 0:
                si[i] = si[i] - sij[i][i + 1]
            elif i == 4:
                si[i] = si[i] - sij[i - 1][i]
            else:
                si[i] = si[i] - sij[i - 1][i] - sij[i][i + 1]

        return si, sij

    def plot(self):
        # A = 1
        epsi = [0.05, 0.05, 0.05, 0.01, 0.1]
        ci = [900, 900, 900, 840, 520]
        # QR = [0, A*(20+3*np.sin(t/4)), 0, 0, 0]
        Lambda = [240, 240, 119, 10.5]
        c0 = 5.67
        kij = np.zeros(4)
        for i in range(4):
            kij[i] = Lambda[i] * self.sij[i][i + 1]
        #初始值
        #init = 20.0, 30.0, 20.0, 30.0, 20.0
        init=20,10,10,20,50
        # init=20,10,10,20,23
        # init=1,1,1,1,1
        t = np.linspace(0, 20, 10001)
        #时间
        sol = odeint(sys_of_funcs, init, t, args=(ci, kij, epsi, self.si))

        # print(sol)
        plt.grid()
        plt.xlabel('t')
        plt.ylabel('T')
        plt.plot(t, sol[:, 0], color='b', label=r"$T_1$")
        plt.plot(t, sol[:, 1], color='c', label=r"$T_2$")
        plt.plot(t, sol[:, 2], color='r', label=r"$T_3$")
        plt.plot(t, sol[:, 3], color='g', label=r"$T_4$")
        plt.plot(t, sol[:, 4], color='m', label=r"$T_5$")

        plt.legend(loc="best")
        plt.show()

        dataframe2 = pd.DataFrame(
            {'t': t, 'T1': sol[:, 0], 'T2': sol[:, 1], 'T3': sol[:, 2], 'T4': sol[:, 3], 'T5': sol[:, 4]})
        dataframe2.to_csv("Results.csv", index=False, sep=',')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec()