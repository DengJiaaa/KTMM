import numpy as np


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

        s = (a + b + c) / 2  # 计算半周长
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))  # 使用海伦公式计算面积

        return area

    def count(self):
        si = np.zeros(5)
        sij = np.zeros((5, 5))
        for i in range(5):
            x = []
            y = []
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
            print(min, max, i, np.min(z), np.max(z))
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

mesh = Mesh("model1.obj")
print(mesh.si, mesh.sij)
