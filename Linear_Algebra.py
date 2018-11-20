#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 00:14:47 2018

Linear Algebra Tutorial (Jupyter notebook)

@author: saul
"""

from __future__ import division, print_function, unicode_literals
import sys
import numpy as np
import numpy.linalg as LA
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D

print("Python version: {}.{}.{}".format(*sys.version_info))
print("Numpy version:", np.version.version)


###~~ VECTORS ~~###
video = np.array([10.5, 5.2, 3.25, 7.0])
video.size
video[2] #3rd element

# 2D vector plot
u = np.array([2,5])
v = np.array([3,1])

x_coords, y_coords = zip(u,v)
plt.scatter(x_coords, y_coords, color=["r","b"])
plt.axis([0,9,0,6])
plt.grid()
plt.show()

# 2D vector arrow plot
def plot_vector2D(vector2D, origin=[0,0], **options):
    return plt.arrow(origin[0], origin[1], vector2D[0], vector2D[1], 
           head_width=0.2, head_length=0.3, length_includes_head=True, 
           **options)

plot_vector2D(u, color="r")
plot_vector2D(v, color="b")
plt.axis([0,9,0,6])
plt.grid()
plt.show()

# 3D vectors
a = np.array([1,2,8])
b = np.array([5,6,3])

# subplot3d = plt.subplot(111, projection='3d')
# x_coords, y_coords, z_coords = zip(a,b)
# subplot3d.scatter(x_coords, y_coords, z_coords)
# subplot3d.set_zlim3d([0, 9])
# plt.show()

def plot_vectors3d(ax, vectors3d, z0, **options):
    for v in vectors3d:
        x, y, z = v
        ax.plot([x,x], [y,y], [z0, z], color="gray", 
                linestyle='dotted', marker=".")
    x_coords, y_coords, z_coords = zip(*vectors3d) # zip element-wise
    ax.scatter(x_coords, y_coords, z_coords, **options)

subplot3d = plt.subplot(111, projection='3d')
subplot3d.set_zlim([0, 9])
plot_vectors3d(subplot3d, [a,b], 5, color=("r","b"))
plt.show()

# Vector Norms (Normalizing vectors via Euclidean norm)
# square and sum up every element in the vector, then square-root = NORM
# essentially the MAGNITUDE or ABOSULUTE LENGTH of the vector
def vector_norm(vector):
    squares = [element**2 for element in vector]
    return sum(squares)**0.5

print("||",u,"|| =")
vector_norm(u)

# Easy way
LA.norm(u)

# Plot
radius = LA.norm(u)
plt.gca().add_artist(plt.Circle((0,0), radius, color="#DDDDFF"))
plot_vector2D(u, color="red")
plt.axis([0,8.7,0,6])
plt.grid()
plt.show()

# VECTOR ADDITION
print(" ", u)
print("+", v)
print("-"*9)
print(" ", u+v)

# Plot
plot_vector2D(u, color="r")
plot_vector2D(v, color="b")
plot_vector2D(v, origin=u, color="b", linestyle="dotted")
plot_vector2D(u, origin=v, color="r", linestyle="dotted")
plot_vector2D(u+v, color="g")
plt.axis([0, 9, 0, 7])
plt.text(0.7, 3, "u", color="r", fontsize=18)
plt.text(4, 3, "u", color="r", fontsize=18)
plt.text(1.8, 0.2, "v", color="b", fontsize=18)
plt.text(3.1, 5.6, "v", color="b", fontsize=18)
plt.text(2.4, 2.5, "u+v", color="g", fontsize=18)
plt.grid()
plt.show()

# Vector addition is COMMUTATIVE => [u + v] = [v + u]
# ie. both paths lead to the same point whichever order they are added
# NB: vector addition is also ASSOCIATIVE => u + [v + w] = [u + v] + w

# GEOMETRIC TRANSLATION
t1 = np.array([2, 0.25])
t2 = np.array([2.5, 3.5])
t3 = np.array([1, 2])

x_coords, y_coords = zip(t1, t2, t3, t1)
plt.plot(x_coords, y_coords, "c--", x_coords, y_coords, "co")
plot_vector2D(v, t1, color="r", linestyle=":")
plot_vector2D(v, t2, color="r", linestyle=":")
plot_vector2D(v, t3, color="r", linestyle=":")
plt.axis([0, 6, 0, 5])

t1b = t1 + v
t2b = t2 + v
t3b = t3 + v

x_coords_b, y_coords_b = zip(t1b, t2b, t3b, t1b)
plt.plot(x_coords_b, y_coords_b, "b-", x_coords_b, y_coords_b, "bo")
plt.text(4, 4.2, "v", color="r", fontsize=18)
plt.text(3, 2.3, "v", color="r", fontsize=18)
plt.text(3.5, 0.4, "v", color="r", fontsize=18)
plt.grid()
plt.show()

# SCALAR MULTIPLICATION
# Every element is multiplied by the scalar
print("1.5 *", u, "="); print(1.5 * u)

k = 2.5 # Scale up by factor x2.5
t1c = k * t1
t2c = k * t2
t3c = k * t3

plt.plot(x_coords, y_coords, "c--", x_coords, y_coords, "co")
plot_vector2D(t1, color="r")
plot_vector2D(t2, color="r")
plot_vector2D(t3, color="r")
x_coords_c, y_coords_c = zip(t1c, t2c, t3c, t1c)
plt.plot(x_coords_c, y_coords_c, "b-", x_coords_c, y_coords_c, "bo")
plot_vector2D(k * t1, color="b", linestyle=":")
plot_vector2D(k * t2, color="b", linestyle=":")
plot_vector2D(k * t3, color="b", linestyle=":")
plt.axis([0, 9, 0, 9])
plt.grid()
plt.show()

# Scalar multiplication is DISTRIBUTIVE over addition of vectors =>
# k * [u + v] = [k * u] + [k * v]

# ZERO vectors are vectors full of zeros
# UNIT vectors have a norm equal to 1
# The NORMALIZED vector of 'u' is the UNIT vector that points in the 
# same direction as 'u'

plt.gca().add_artist(plt.Circle((0,0),1,color='c'))
plt.plot(0, 0, "ko")
plot_vector2D(v / LA.norm(v), color="k")
plot_vector2D(v, color="b", linestyle=":")
plt.text(0.3, 0.3, "$\hat{u}$", color="k", fontsize=18)
plt.text(1.5, 0.7, "$u$", color="b", fontsize=18)
plt.axis([-1.5, 5.5, -1.5, 3.5])
plt.grid()
plt.show()

# DOT PRODUCT
# u * v = ||u|| * ||v|| * cos(w)   where, w = angle between u and v
# 
# ie. multiplication element-wise, then calculate sum total
def dot_product(v1, v2):
    return sum(v1i * v2i for v1i, v2i in zip(v1, v2))

dot_product(u, v)

# Easy way
np.dot(u, v)
u.dot(v)

# The Dot Product is COMMUTATIVE, and ASSOCIATIVE w.r.t. scalar 
# multiplication and is DISTRIBUTIVE over addition of vectors
# The Dot Product is useful to project points onto an axis, ie. v onto u

def vector_angle(u, v):
    cos_theta = u.dot(v) / LA.norm(u) / LA.norm(v)
    return np.arccos(np.clip(cos_theta, -1, 1))

theta = vector_angle(u, v)
print("Angle =", theta, "radians")
print("      =", theta * 180 / np.pi, " degrees")

# If the dot product of two non-null vectors is zero,
# the two vectors are ORTHOGONAL

# VECTOR PROJECTION
u_normalized = u / LA.norm(u)
# 2 equivalent formulae:
# proj_v_onto_u = (u.dot(v) / LA.norm(u)**2) * u
proj_v_onto_u = v.dot(u_normalized) * u_normalized

plot_vector2D(u, color="r")
plot_vector2D(v, color="b")
plot_vector2D(proj_v_onto_u, color="k", linestyle=":")
plt.plot(proj_v_onto_u[0], proj_v_onto_u[1], "ko")
plt.plot([proj_v_onto_u[0], v[0]], [proj_v_onto_u[1], v[1]], "b:")
plt.text(1, 2, "$proj_u v$", color="k", fontsize=18)
plt.text(1.8, 0.2, "$v$", color="b", fontsize=18)
plt.text(0.8, 3, "$u$", color="r", fontsize=18)
plt.axis([0, 8, 0, 5.5])
plt.grid()
plt.show()


##~~ MATRICES ~~##
# List of lists in a numpy array
A = np.array([[10,20,30],[40,50,60]]) # Uppercase variable name by convention
A.size # total no. elements in matrix, not its dimensions
A.shape # get dimensions
A[1,:]

# To access position A(2,3) we write:
A[1,2] # 2nd row, 3rd column
A[1,:] # 2nd row as a 1D vector
# Slice to access rows or columns
A[:,0:2] # EXCLUSIVE BEWARE! Excludes 3rd row! +1 to be inclusive

# Types of matrices:
# square            -> n_rows = n_cols
# upper trianglular -> square matrix where all elements below diagonal = 0
# diagonal          -> both upper and lower triangular matrix 
np.diag([4,5,6]) # (ie. only the diagonal contains non-zero values)
# identity          -> diagonal of 1's, size n x n
np.eye(3)
# Matrices may be linearly transformed, ie. by rotation, translation, or scaling

# DIAGONAL MATRICES
np.diag([4,5,6])

D = np.array(np.reshape(list(range(1,10)),(3,3)))
np.diag(D) # Works both ways
# IDENTITY MATRICES
np.eye(3)

# MATRIX ADDITION
B = np.array([[1,2,3],[4,5,6]])
B
A
A + B 
B + A # Element-wise, COMMUTATIVE

C = np.array([[100,200,300],[400,500,600]])
A + (B + C)
B + (C + A) # Also ASSOCIATIVE

# MATRIX-SCALAR MULTIPLICATION
2 * A
A * 2 # COMMUTATIVE
2 * (3 * A)
(2 * 3) * A # ASSOCIATIVE
2 * (A + B)
2 * A + 2 * B # DISTRIBUTIVE (over addition of matrices)

# MATRIX MULTIPLICATION
D = np.array([[2,3,5,7],[11,13,17,19],[23,29,31,37]])
A
E = A.dot(D)
print(D,'\n\n',A,'\n\n',E)
40*5 + 50*17 + 60*31 # 2nd row, 3rd col in E
E[1,2]

10*2 + 20*11 + 30*23
E[0,0]

40*7 + 50*19 + 60*37
E[1,3]

10*5 + 20*17 + 30*31
E[0,2]
# Sussed matrix multiplication!

try:
    D.dot(A)
except ValueError as e:
    print("ValueError:", e)
# Matrix multiplication is NOT COMMUTATIVE (QR != RQ)

F = np.array([[5,2],[4,1],[9,3]])
A
E1 = A.dot(F)
# By the rows of A, columns of F (2x2)

5*10 + 4*20 + 9*30
E1[0,0]

2*40 + 1*50 + 3*60
E1[1,1]

# Stil got it!

E2 = F.dot(A)
# This time by the rows of F, columns of A (3x3)

10*5 + 40*2
E2[0,0]

20*5 + 50*2
E2[0,1]

30*9 + 60*3
E2[2,2]
# BOOM!

# Matrix multiplication is ASSOCIATIVE
G = np.array([[8,7,4,2,5],[2,5,1,0,5],[9,11,17,21,0],[0,1,0,1,2]])
A.dot(D).dot(G)
A.dot(D.dot(G))

E3 = A.dot(D)
10*2 + 20*11 + 30*23
E3[0,0]

E4 = E3.dot(G)
930*8 + 1160*2 + 1320*9 + 1560*0
E4[0,0]

E5 = D.dot(G)
2*8 + 3*2 + 5*9 + 7*0
E5[0,0]

E6 = A.dot(E5)
10*67 + 20*267 + 30*521
E6[0,0]

# Matrix multiplication is also DISTRIBUTIVE over addition of matrices
(A + B).dot(D)
A.dot(D) + B.dot(D)

# Identity Matrices: MI = IM = M 
# Any matrix, multiplied by an identity matrix (of equal size), is equal to itself
A
A.dot(np.eye(3))
np.eye(2).dot(A)

# CAUTION: Use A.dot(B) and not A*B for matrix multiplication 
A * B # NOT a matrix multiplication, but an element-wise product

# MATRIX TRANSPOSE
# Flips the rows and columns
A
A.T
A.T.T 
# Transposition is DISTRIBUTIVE of addition of matrices
(A + B).T
A.T + B.T

A
D
(A.dot(D)).T
D.T.dot(A.T)

# Symmetric Matrices are equal to their transposes
# M.T = M
SYM = D.dot(D.T)
SYM.T
# The product of a matrix by its transpose is always a symmetric matrix

# Converting 1D arrays to 2D arrays in NumPy
u
u.T
u_row = np.array([u])
u_row
u[np.newaxis, :] # This is a 2D array with just 1 row, quite explicit
u[None] # Equivalent, but a little less explicit
# Now we can transpose our vector into a column vector
u_row.T

# Plotting a Matrix
P = np.array([[3.0,4.0,1.0,4.6],[0.2,3.5,2.0,0.5]])
x_coords_P, y_coords_P = P
plt.scatter(x_coords_P, y_coords_P)
plt.axis([0,5,0,4])
plt.show()

# Since the vectors are ordered like a list in the matrix
# the points may be connected as a path
plt.plot(x_coords_P, y_coords_P, "bo")
plt.plot(x_coords_P, y_coords_P, "b--")
plt.axis([0,5,0,4])
plt.grid()
plt.show()

plt.gca().add_artist(Polygon(P.T))
plt.axis([0,5,0,4])
plt.grid()
plt.show()

# GEOMETRIC MATRIX OPERATIONS
# Vector addtion results in geometric translation
# Vector multiplication by scalar results in resizing/zooming, centred on origin
# Vector dot product results in projecting a vector onto another vector (coords)

# MATRIX ADDITION
# Adding two matrices together is equivalent to adding all their vectors together
H = np.array([[0.5,-0.2,0.2,-0.1],[0.4,0.4,1.5,0.6]])
P_moved = P + H

plt.gca().add_artist(Polygon(P.T, alpha=0.2))
plt.gca().add_artist(Polygon(P_moved.T, alpha=0.3, color="r"))
for vector, origin in zip(H.T, P.T): # Zip up vectors (rows) together
    plot_vector2D(vector, origin=origin)

plt.text(2.2, 1.8, "$P$", color="b", fontsize=18)
plt.text(2.0, 3.2, "$P+H$", color="r", fontsize=18)
plt.text(2.5, 0.5, "$H_{*,1}$", color="k", fontsize=18)
plt.text(4.1, 3.5, "$H_{*,2}$", color="k", fontsize=18)
plt.text(0.4, 2.6, "$H_{*,3}$", color="k", fontsize=18)
plt.text(4.4, 0.2, "$H_{*,4}$", color="k", fontsize=18)
plt.axis([0,5,0,4])
plt.grid()
plt.show()

# Adding a matrix full of identical vectors results in simple geometric translation
H2 = np.array([[-0.5,-0.5,-0.5,-0.5],[0.4,0.4,0.4,0.4]])
P_translated = P + H2

plt.gca().add_artist(Polygon(P.T, alpha=0.2))
plt.gca().add_artist(Polygon(P_translated.T, alpha=0.3, color="r"))
for vector, origin in zip(H2.T, P.T):
    plot_vector2D(vector, origin=origin)
plt.axis([0,5,0,4])
plt.grid()
plt.show()

# Matrices can only be added together if they have the same dimensions
# But NumPy allows adding a row or column vector to a matrix by BROADCASTING

# BROADCASTING WITH NUMPY
P + [[-0.5],[0.4]] # Broadcasted across the matrix, recycled to fill n_rows or n_cols

# SCALAR MULTIPLICATION
def plot_transformation(P_before, P_after, text_before, text_after, 
                        axis = [0,5,0,4], arrows=False):
    if arrows:
        for vector_before, vector_after in zip(P_before.T, P_after.T):
            plot_vector2D(vector_before, color="blue", linestyle="--")
            plot_vector2D(vector_after, color="red", linestyle="-")
    plt.gca().add_artist(Polygon(P_before.T, alpha=0.2))
    plt.gca().add_artist(Polygon(P_after.T, alpha=0.3, color="r"))
    plt.text(P_before[0].mean(), P_before[1].mean(), text_before, 
             fontsize=18, color="blue")
    plt.text(P_after[0].mean(), P_after[1].mean(), text_after, 
             fontsize=18, color="red")
    plt.axis(axis)
    plt.grid()

P_rescaled = 0.60 * P
plot_transformation(P, P_rescaled, "$P$", "$0.6 P$", arrows=True)
plt.show()

# MATRIX MULTIPLICATION - PROJECTION ONTO AN AXIS
U = np.array([[1,0]]) # Basically just a horizontal unit vector
U.dot(P) # This just returns the horizontal coords of the vectors in P
# In other words, we just projected P onto the horizontal axis

def plot_projection(U, P):
    U_P = U.dot(P)
    axis_end = 100 * U
    plot_vector2D(axis_end[0], color="black")
    plt.gca().add_artist(Polygon(P.T, alpha=0.2))
    for vector, proj_coord in zip(P.T, U_P.T):
        proj_point = proj_coord * U
        plt.plot(proj_point[0][0], proj_point[0][1], "ro")
        plt.plot([vector[0], proj_point[0][0]], [vector[1], proj_point[0][1]], "r--")
    plt.axis([0,5,0,4])
    plt.grid()
    plt.show()

plot_projection(U, P)

# You may project onto any other axis by switching U with some other unit vector
angle30 = 30 * np.pi / 180 # 30 degree angle (in radians)
U_30 = np.array([[np.cos(angle30), np.sin(angle30)]])

plot_projection(U_30, P)
# REMEMBER that the DOT PRODUCT of a UNIT VECTOR and a MATRIX is just a
# PROJECTION on an AXIS => Gives coords of resulting points along that axis

# MATRIX MULTIPLICATION - ROTATION
# Create a 2x2 matrix containing 2 unit vectors that make 
# 30 and 120 degree angles with the horizontal axis
angle120 = 120 * np.pi / 180
V = np.array([[np.cos(angle30), np.sin(angle30)],[np.cos(angle120), np.sin(angle120)]])
V.dot(P) # V is a ROTATION MATRIX

# The DOT PRODUCT of V on P will rotate matrix P about the origin
P_rotated = V.dot(P) 
plot_transformation(P, P_rotated, "$P$", "$VP$", [-2,6,-2,4], arrows=True)
plt.show()

# MATRIX MULTIPLICATION - OTHER LINEAR TRANSFORMATIONS
# In general, the matrix on left side of dot product specifies what linear 
# transformation to apply to the vectors on the right side 
# This can be used for projections, rotations, shear and squeeze mapping

# SHEAR MAPPING
F_shear = np.array([[1,1.5],[0,1]])
plot_transformation(P, F_shear.dot(P), "$P$", "$F_{shear} P$", axis=[0,10,0,7])
plt.show()
# Easier to see the effect of a shear on a unit square
Square = np.array([[0,0,1,1],[0,1,1,0]])
plot_transformation(Square, F_shear.dot(Square), "$Square$", "$F_{shear} Square$", axis=[0,2.6,0,1.8])

# SQUEEZE MAPPING
F_squeeze = np.array([[1.4,0],[0,1/1.4]])
plot_transformation(P, F_squeeze.dot(P), "$P$", "$F_squeeze P$", axis=[0,7,0,5])
plt.show()
# Effect on a unit square
plot_transformation(Square, F_squeeze.dot(Square), "$Square$", "$F_{squeeze} Square$", axis=[0,1.8,0,1.2])
plt.show()

# REFLECTION IN THE HORIZONTAL AXIS
F_reflect = np.array([[1,0],[0,-1]])
plot_transformation(P, F_reflect.dot(P), "$P$", "$F_{reflect} P$", axis=[-2,9,-4.5,4.5])
plt.show()

# MATRIX INVERSE
# Given that matrices can represent any linear transformation, can we find 
# a matrix that reverses the effect of a given transformation matrix F?
F_inv_shear = np.array([[1,-1.5],[0,1]])
P_sheared = F_shear.dot(P)
P_unsheared = F_inv_shear.dot(P_sheared) # Inverse transformation
plot_transformation(P_sheared, P_unsheared, "$P_{sheared}$", "$P_{unsheared}$", axis=[0,10,0,7])
plt.plot(P[0], P[1], "b--")
plt.show()

# NumPy inverse function
F_inv_shear = LA.inv(F_shear)
F_inv_shear
# Only square matrices can be inversed
# If you use a 2x3 matrix to project a 3D object onto a plane, 
# the result is lost information that cannot be retrieved
plt.plot([0, 0, 1, 1, 0, 0.1, 0.1, 0, 0.1, 1.1, 1.0, 1.1, 1.1, 1.0, 1.1, 0.1],
         [0, 1, 1, 0, 0, 0.1, 1.1, 1.0, 1.1, 1.1, 1.0, 1.1, 0.1, 0, 0.1, 0.1],
         "r-")
plt.axis([-0.5, 2.1, -0.5, 1.5])
plt.show()
# It is impossible to tell whether this is the projection of a perfect cube,
# or a narrow rectangular object

# SINGULAR MATRICES (aka. DEGENERATE MATRICES)
# Even square transformation matrices can lose information
F_project = np.array([[1,0],[0,0]])
plot_transformation(P, F_project.dot(P), "$P$", "$F_{project} \cdot P$", axis=[0,6,-1,4])
plt.show()
# Here F_prpject has no inverse, you cannot go back to the original polygon
# A square matrix that has no computable inverse is called a SINGULAR MATRIX
try:
    LA.inv(F_project)
except LA.LinAlgError as e:
    print("LinAlgError:", e)

angle30 = 30 * np.pi / 180
F_project_30 = np.array([[np.cos(angle30)**2, np.sin(2 * angle30)/2],
                         [np.sin(2 * angle30)/2, np.sin(angle30)**2]])
plot_transformation(P, F_project_30.dot(P), "$P$", "$F_{project\_30} \cdot P$", axis=[0,6,-1,4])
plt.show()
# NumPy can sometimes compute the inverse of a singular matrix 
# due to floating point rounding errors (notice the magnitude of the elements!)
LA.inv(F_project_30)

# The DOT PRODUCT of a MATRIX BY ITS INVERSE results in an IDENTITY MATRIX
F_shear.dot(LA.inv(F_shear))
np.eye(2)
# ie. the inverse of the inverse gives back the matrix itself
LA.inv(LA.inv(F_shear))
F_shear

# INVOLUTION: A matrix that is ITS OWN INVERSE
# eg. reflection matrices or rotation about 180 degrees

# A COMPLEX INVOLUTION: horizontal squeeze, followed by reflection in y, then 90 degree rotation
F_involution = np.array([[0,-2],[-1/2,0]])
plot_transformation(P, F_involution.dot(P), "$P$", "$F_{involution} \cdot P$", axis=[-8,5,-4,4])
plt.show()

# ORTHOGONAL MATRIX: A SQUARE MATRIX, whose INVERSE is its own TRANSPOSE
#            H^-1 = H.T
# therefore: H * H.T = H.T * H = I
F_reflect.dot(F_reflect.T)

# DETERMINANT







