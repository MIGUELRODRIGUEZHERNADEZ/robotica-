
import sympy as sp

sp.init_printing()  # Para que se vea bonito en Jupyter


# Ejercicio 1: Suma de matrices

A1 = sp.Matrix([[2, 4, 6],
                [1, 3, 5],
                [7, 9, 11]])

B1 = sp.Matrix([[12, 10, 8],
                [6, 4, 2],
                [0, -2, -4]])

C1 = A1 + B1

print("Ejercicio 1: Suma de matrices")
print("A + B =")
sp.pprint(C1)
print("\n")


# --------------------------------------------
# Ejercicio 2: Multiplicación de matrices
# --------------------------------------------
A2 = sp.Matrix([[2, 1],
                [3, 4],
                [5, 6]])

B2 = sp.Matrix([[7, 8],
                [9, 10]])

C2 = A2 * B2

print("Ejercicio 2: Multiplicación de matrices")
print("A * B =")
sp.pprint(C2)
print("\n")


# --------------------------------------------
# Ejercicio 3: Inversión de matriz
# --------------------------------------------
A3 = sp.Matrix([[4, 7, 2],
                [2, 6, 8],
                [3, 1, 9]])

A3_inv = A3.inv()

print("Ejercicio 3: Inversa de la matriz A")
print("A =")
sp.pprint(A3)
print("\nA^(-1) =")
sp.pprint(A3_inv)
print("\n")


# --------------------------------------------
# Ejercicio 4: Sistema de ecuaciones AX = B
# --------------------------------------------
# Sistema:
# 2x +  y +  z =  8
# 3x + 5y + 2z = 21
#  x + 2y + 4z = 11

A4 = sp.Matrix([[2, 1, 1],
                [3, 5, 2],
                [1, 2, 4]])

B4 = sp.Matrix([8, 21, 11])

X = A4.LUsolve(B4)   # Resuelve AX = B

print("Ejercicio 4: Resolución de sistema lineal AX = B")
print("Matriz A =")
sp.pprint(A4)
print("\nVector B =")
sp.pprint(B4)
print("\nSolución X = [x, y, z]^T =")
sp.pprint(X)
print("\n")


# --------------------------------------------
# Ejercicio 5: Determinante de una matriz
# --------------------------------------------
A5 = sp.Matrix([[3, -2, 1],
                [0, 5, 4],
                [2, 1, 7]])

det_A5 = A5.det()

print("Ejercicio 5: Determinante de A")
print("A =")
sp.pprint(A5)
print("\nDeterminante det(A) = ", det_A5)
print("\n")


# --------------------------------------------
# Ejercicio 6: Producto cruz de vectores
# --------------------------------------------
A6 = sp.Matrix([2, 3, -1])
B6 = sp.Matrix([1, -2, 4])

cross_AB = A6.cross(B6)

print("Ejercicio 6: Producto cruz A x B")
print("A =", A6)
print("B =", B6)
print("A x B =")
sp.pprint(cross_AB)
print("\n")


# --------------------------------------------
# Ejercicio 7: Proyección ortogonal
# Proyección de V sobre U: proj_U(V) = (V·U / U·U) * U
# --------------------------------------------
V7 = sp.Matrix([5, -3, 2])
U7 = sp.Matrix([2, 1, 2])

proj_V_on_U = (V7.dot(U7) / U7.dot(U7)) * U7

print("Ejercicio 7: Proyección ortogonal de V sobre U")
print("V =", V7)
print("U =", U7)
print("proj_U(V) =")
sp.pprint(proj_V_on_U)
print("\n")


# --------------------------------------------
# Ejercicio 8: Producto escalar de proyecciones
# proj_U(V) · proj_W(V)
# --------------------------------------------
V8 = sp.Matrix([3, -1, 2])
U8 = sp.Matrix([2, 2, -1])
W8 = sp.Matrix([1, 4, -2])

proj_V_on_U8 = (V8.dot(U8) / U8.dot(U8)) * U8
proj_V_on_W8 = (V8.dot(W8) / W8.dot(W8)) * W8
dot_proj = proj_V_on_U8.dot(proj_V_on_W8)

print("Ejercicio 8: Producto escalar de proyecciones")
print("V =", V8)
print("U =", U8)
print("W =", W8)
print("proj_U(V) =")
sp.pprint(proj_V_on_U8)
print("\nproj_W(V) =")
sp.pprint(proj_V_on_W8)
print("\nProducto escalar proj_U(V) · proj_W(V) =")
sp.pprint(dot_proj)
print("\n")


# --------------------------------------------
# Ejercicio 9: Ortogonalización de Gram-Schmidt
# --------------------------------------------
v1 = sp.Matrix([1, 1, 0])
v2 = sp.Matrix([1, 2, 1])
v3 = sp.Matrix([2, 1, 3])

def gram_schmidt(vs):
    us = []
    for v in vs:
        u = v
        for prev in us:
            u = u - (v.dot(prev) / prev.dot(prev)) * prev
        us.append(sp.simplify(u))
    return us

u1, u2, u3 = gram_schmidt([v1, v2, v3])

# Podemos reescalar para tener vectores más bonitos
u1_s = u1               # [1, 1, 0]
u2_s = 2 * u2           # multiplicamos por 2
u3_s = (3/sp.Integer(4)) * u3   # para obtener enteros

print("Ejercicio 9: Gram-Schmidt")
print("Vectores originales:")
print("v1 =", v1)
print("v2 =", v2)
print("v3 =", v3)
print("\nVectores ortogonales (sin reescalar):")
print("u1 =")
sp.pprint(u1)
print("u2 =")
sp.pprint(u2)
print("u3 =")
sp.pprint(u3)
print("\nUna versión entera y ortogonal equivalente:")
print("u1' =")
sp.pprint(u1_s)   # [1, 1, 0]
print("u2' =")
sp.pprint(u2_s)   # [-1, 1, 2]
print("u3' =")
sp.pprint(u3_s)   # [1, -1, 1]
print("\n")


# --------------------------------------------
# Ejercicio 10: Espacio nulo de A
# A ya está en forma escalonada reducida.
# --------------------------------------------
A10 = sp.Matrix([[1, 2, 0, 3],
                 [0, 1, 0, 2],
                 [0, 0, 1, 1]])

null_space = A10.nullspace()

print("Ejercicio 10: Espacio nulo de A")
print("A =")
sp.pprint(A10)
print("\nBase del espacio nulo (vectores v tales que A v = 0):")
for v in null_space:
    sp.pprint(v)
print("\n")
