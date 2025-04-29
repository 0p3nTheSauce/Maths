#Single head attention
from lin_al import softmax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def disp_matrix(mat, title, xlabel, ylabel):
  """
  Display a matrix using seaborn heatmap
  """
  plt.figure(figsize=(10, 8))
  sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlGnBu")
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  


def attention(Q, K, V):
  """
  Q: query matrix
  K: key matrix
  V: value matrix
  """
  d = Q.shape[1] #dimension of vector
  attention_weights = softmax(Q @ K.T / np.sqrt(d)) #scaled dot product 
  disp_matrix(attention_weights, "Attention Weights", "Key", "Query")
  weighted_sum = attention_weights @ V #weighted sum of value vectors
  return weighted_sum
  
def main():
  dmodel = 10
  dk = dv = dmodel
  n = 15
  E = np.random.rand(n, dmodel) #embedding matrix
  W_Q = np.random.rand(dmodel, dk) #query weight matrix
  W_K = np.random.rand(dmodel, dk) #key weight matrix
  W_V = np.random.rand(dmodel, dv) #value weight matrix
  Q = E @ W_Q #query matrix
  K = E @ W_K #key matrix
  V = E @ W_V #value matrix
  disp_matrix(Q, "Query Matrix", "Query", "i")
  disp_matrix(K, "Key Matrix", "Key", "i")
  disp_matrix(V, "Value Matrix", "Value", "i")
  output = attention(Q, K, V)
  plt.figure(figsize=(10, 8))
  sns.heatmap(E, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=[])
  plt.title("Embedding Matrix")
  plt.xlabel("Embedding")
  plt.ylabel("i")
  plt.show()
  
  plt.figure(figsize=(10, 8))
  sns.heatmap(output, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=[])
  plt.title("Output Matrix")
  plt.xlabel("Output")
  plt.ylabel("i")
  plt.show()
  
  E_updated = E + output
  plt.figure(figsize=(10, 8))
  sns.heatmap(E_updated, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=[])
  plt.title("Updated Embedding Matrix")
  plt.xlabel("Embedding")
  plt.ylabel("i")
  plt.show()
  
  
  
  

if __name__ == "__main__":
  main()
  