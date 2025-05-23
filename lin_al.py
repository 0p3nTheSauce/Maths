import numpy as np 
import time
import matplotlib.pyplot as plt
import seaborn as sns
def my_dot(a, b):
  '''Hand made dot product function'''
  a_rows, a_cols = a.shape
  b_rows, b_cols = b.shape
  C = []
  #check if the matrices are well formed
  if any([len(row) != len(a[0]) for row in a]):
    raise ValueError("A is not well formed")
  if any([len(row) != len(b[0]) for row in b]):
    raise ValueError("A is not well formed")
  if a_rows != b_cols:
    raise ValueError("Incompatible shapes")
  # perform dot product
  for row_a in range(a_rows):
    new_row = []
    for col_b in range(b_cols):
      new_row.append(sum([a[row_a,col_a] * b[col_a,col_b] for col_a in range(a_cols)]))
    C.append(new_row)
  return np.array(C)

def polkadots(A, B):
  '''This function tests if my hand made function is equivalent to np.dot()'''
  C = np.dot(A, B)
  print(C)

  D = my_dot(A, B)
  print(D)
  # print(A*B)
  
def low_rank_approx():
  '''
  This function tests the computational complexity of the dot product between a large 
  (high dimensional) matrix, and two smaller matrices.
  Time taken for X @ A @ B: 0.266175 seconds
  Time taken for X @ A @ B: 1.288740 seconds
  an order of magnitude slower
  '''
  m, n, k = 1000, 800, 100  # Large matrix W of shape (1000, 800), approximated by (1000, 100) x (100, 800)

  A = np.random.rand(m, k)  # Smaller matrix
  B = np.random.rand(k, n)
  W = np.random.rand(1000,800)
  # Approximate W
  W_approx = np.dot(A, B)
  
  # X shape: (batch_size, 1000)
  X = np.random.rand(32, m)

  # Efficient multiplication
  start = time.perf_counter()
  for _ in range(1000):
    result = X @ A @ B  # Instead of X @ W
  end = time.perf_counter()
  print(f"Time taken for X @ A @ B: {end - start:.6f} seconds")
  # in-efficient multiplication
  start = time.perf_counter()
  for _ in range(1000):
    result = X @ W  
  end = time.perf_counter()
  print(f"Time taken for X @ A @ B: {end - start:.6f} seconds")
  
def sum_vectors(q_vectors, k_vectors):
  if len(q_vectors) != len(k_vectors):
    raise ValueError("must have same number of vectors in a and b")
  acc = 0
  for i in range(len(q_vectors)):
    acc += np.dot(q_vectors[i], k_vectors[i])
  print(f"sum of vector pair dot products: {acc}")

  total = np.sum(q_vectors * k_vectors)  # element-wise multiply, then sum
  #timing not really fair because pure python vs numpy
  print(f"sum of vector pair dot products: {total}")  
  
  
def dot_products_vectors(q_vectors,k_vectors):
  '''This function shows that if we have a sequenc of vectors a0 -> ai, 
  and a sequence of vectors b0 -> bi, and we want to compute the dot product 
  between every pair of vectors from both sequences, we can do so efficiently by 
  stacking the vectors, and performing the dot product of matrices A and the transpose of B.'''
  if len(q_vectors) != len(k_vectors):
    raise ValueError("must have same number of vectors in a and b")
  vector_dot_matrix = np.ones_like(q_vectors)
  for i in range(len(q_vectors)):
    for j in range(len(k_vectors)):
      vector_dot_matrix[i,j] = np.dot(q_vectors[i], k_vectors[j])
  vector_dot_matrix = np.array(vector_dot_matrix)
  print(vector_dot_matrix)
  stacked_dot_matrix = q_vectors @ k_vectors.T
  print(stacked_dot_matrix)
  
def my_softmax(vec):
  '''my own implementation of soft max. Not as good as the datacamp one,
  because I can just write softmax(mat) on a matrix and have it just work'''
  max_val = max(vec)
  vec = [val-max_val for val in vec]#subtract the max values for numerical stability
  denominator = sum([np.exp(val) for val in vec]) #sum of e^value for all values
  return np.array([np.exp(elem)/denominator for elem in vec]) #soft max output for value = e^value/denominator  

def softmax(x):
    """
    some code i found online to check that I wrote softmax correctly
    Compute softmax values for each set of scores in x.
    
    Args:
        x: Input array of shape (batch_size, num_classes) or (num_classes,)
        
    Returns:
        Softmax probabilities of same shape as input
    """
    # For numerical stability, subtract the maximum value from each input vector
    # This prevents overflow when calculating exp(x)
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    
    # Calculate exp(x) for each element
    exp_x = np.exp(shifted_x)
    
    # Calculate the sum of exp(x) for normalization
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    
    # Normalize to get probabilities
    probabilities = exp_x / sum_exp_x
    
    return probabilities

def test_softmaxes():
  vecs = np.random.rand(7,7)
  print("My softmax function:")
  for vec in vecs:
    print(my_softmax(vec))
  print()
  print("Datacamp softmax:")
  for vec in vecs:
    print(softmax(vec))

  if any([np.any(my_softmax(vec) != softmax(vec)) for vec in vecs]):
    print("Fail")
  else:
    print("Pass")

def attention_pattern():
  '''This function demonstrates how the dot products between every pair of query
    and key vectors shows a correlation pattern'''
  Q = np.random.rand(7,7)
  K = np.random.rand(7,7)
  d = Q.shape[1] #dimension of vector
  attention_weights = softmax(Q @ K.T / np.sqrt(d)) #scaled dot product 
  
  sns.heatmap(attention_weights,annot=True, cmap="YlGnBu")

  plt.title("Attention pattern")
  plt.xlabel("Key vectors")
  plt.ylabel("Query vectors")
  plt.show()
  

def test_shapes():
  A = np.random.rand(2,3) #A projects v to a lower dimension: 3D -> 2D
  B = np.random.rand(3,2) #B projects the output of A @ v, back to a higher dimension (the shape of b is the inverse of A): 2D -> 3D
  C = np.random.rand(3,3) #C has no effect on the dimesnions of v (because C is square)
  v = np.random.rand(3,) #v is a 3D vector

  # Demonstrate matrix operations
  print(f"A: {A.shape}\n{A}\n")
  print(f"B: {B.shape}\n{B}\n")
  print(f"C: {C.shape}\n{C}\n")
  print(f"v: {v.shape}\n{v}\n")

  # Matrix-vector multiplications
  print("Down projection:")
  D = A @ v
  print(f"D = A @ v: {D.shape}\n{D}\n")
  
  print("Up projection:")
  E = B @ D
  print(f"E = B @ D: {E.shape}\n{E}\n")

  # Dimension mismatch example
  print(f"B @ v won't work: B shape {B.shape}, v shape {v.shape}\n")

  print("Linear transformation:")
  G = C @ v
  print(f"G = C @ v: {G.shape}\n{G}")

def weighted_sum_man(e_vectors, v_vectors, weights):
  new_e_vectors = []
  for row, e_vec in enumerate(e_vectors):
    weighted_sum = np.array([0]*len(e_vec), dtype=np.float64)
    #compute weigthed sum of the value vectors
    for col, v_vec in enumerate(v_vectors):
      weighted_sum += weights[row][col] * v_vec
    new_e_vectors.append(weighted_sum+e_vec)
  return np.array(new_e_vectors)

def weighted_sum_mat(e_vectors, v_vectors, weights):
  return e_vectors + (weights @ v_vectors)

def test_weighted_sum():
  Q = np.random.rand(7,7) #query matrix
  K = np.random.rand(7,7) #key matrix
  d = Q.shape[1] #dimension of vector
  attention_weights = softmax(Q @ K.T / np.sqrt(d)) #scaled dot product 
  V = np.random.rand(7,10) #value matrix
  E = np.random.rand(7,10) #embedding matrix
  man = weighted_sum_man(E, V, attention_weights)
  mat = weighted_sum_mat(E, V, attention_weights)
  if np.allclose(man, mat):
    print("Pass")
  else:
    print("Fail")

def linear_projection():
  dmodel = 10
  dk = 7
  n = 15
  e_vectors = np.random.rand(n, 10)
  weight_mat = np.random.rand(dmodel,dk)
  manual_k_vectors = []
  #manual way first 
  for ei in e_vectors: 
    ki = ei @ weight_mat #reversed if column vector
    manual_k_vectors.append(ki)
  manual_k_vectors = np.array(manual_k_vectors)
  #single matrix operation
  mat_k_vectors = e_vectors @  weight_mat
  print("Pass" 
    if np.allclose(manual_k_vectors, mat_k_vectors)
    else "Fail")
  
def stacked_vec_by_mat():
  e = np.random.rand(7,) 
  E = np.stack([e for _ in range(7)]) #embedding matrix
  Wq = np.random.rand(7,7) #query matrix
  Q_man = np.stack([e @ Wq for _ in range(7)]) #query matrix   (The order is not what I would have expected)
  Q_mat = E @ Wq #query matrix
  if np.allclose(Q_man, Q_mat):
    print("Pass")
  else: 
    print("Fail")
    
def main():
  # A = np.array([[1, 2],
  #             [3, 4]])

  # B = np.array([[5, 6],
  #               [7, 8]])
  # sum_vectors(A, B)  
  attention_pattern()
  # test_weighted_sum()
  # linear_projection()
  # test_softmaxes()
  # test_shapes()
  print()
  print("############################################################################")
  print()
  #dot_products_vectors(A, B)
  
if __name__  == "__main__":
  main()