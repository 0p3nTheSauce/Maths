# Create random matrices and vector
A = np.random.rand(2,3)  # Projects 3D -> 2D
B = np.random.rand(3,2)  # Projects 2D -> 3D
C = np.random.rand(3,3)  # Square matrix (preserves dimensions)
v = np.random.rand(3,)   # 3D vector

# Demonstrate matrix operations
print(f"A: {A.shape}\n{A}\n")
print(f"B: {B.shape}\n{B}\n")
print(f"C: {C.shape}\n{C}\n")
print(f"v: {v.shape}\n{v}\n")

# Matrix-vector multiplications
D = A @ v
print(f"D = A @ v: {D.shape}\n{D}\n")

E = B @ D
print(f"E = B @ D: {E.shape}\n{E}\n")

# Dimension mismatch example
print(f"B @ v won't work: B shape {B.shape}, v shape {v.shape}\n")

G = C @ v
print(f"G = C @ v: {G.shape}\n{G}")