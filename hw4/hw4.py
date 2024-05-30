from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

#Create mesh and define function space
R = 1.0
circle_r = Circle(Point(0, 0), R) #задача в круге
mesh = generate_mesh(circle_r, 64)
V = FunctionSpace(mesh, 'P', 1)
bounds = MeshFunction("size_t", mesh, 1)

#ГРАНИЧНЫЕ УСЛОВИЯ
class boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary or (x[0] < 0) #x[1]

b2 = boundary()
b2.mark(bounds, 1)

# test 1
# u = r *r * sin(phi) , R = 1, alpha = 3

r = Expression("sqrt(x[0] * x[0] + x[1] * x[1])", degree = 2)
phi = Expression("atan2(x[1], x[0])", degree = 2)

u_D = Expression("sin(phi)", phi = phi, degree = 2)
alpha = Constant(3)
f = Expression("(3*r*r - 1) * sin(phi)",  r = r, phi = phi, degree = 2)
h = Expression("R*R * sin(phi)", phi = phi, degree = 2)
g = Expression("2*R*sin(phi)", phi = phi, degree = 2)


#ВАРИАЦИОННАЯ ЗАДАЧА
u = TrialFunction(V)
v = TestFunction(V)
bc = DirichletBC(V, h, bounds, 1)
a = dot(grad(u), grad(v))*dx + alpha*u*v*dx
L = f*v*dx - g*v*ds
u = Function(V)
solve(a == L, u, bc)

#НОРМЫ L2 и максимум-норма
# 误差计算
error_L2 = errornorm(u_D, u, 'L2')
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_C = np.max(np.abs(vertex_values_u - vertex_values_u_D))
print('L2-error = ', error_L2)
print('C-error = ', error_C)

#Визуализация
n = mesh.num_vertices()
d = mesh.geometry().dim()
mesh_coordinates = mesh.coordinates().reshape((n, d))
triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
plt.figure()
zfaces = np.asarray([u_D(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.colorbar()
plt.title("Аналитическое решение")
plt.savefig('аналитическое.png')

plt.figure()
zfaces = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.colorbar()
plt.title("Численное решение")
plt.savefig('численное.png')