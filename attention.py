#Single head attention
from lin_al import softmax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def disp_matrix(mat, title, xlabel, ylabel, xticks=True):
  """
  Display a matrix using seaborn heatmap
  """
  plt.figure(figsize=(10, 8))
  if xticks:
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlGnBu")
  else:
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=[])
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
  disp_matrix(E, "Embedding Matrix", "Embedding", "i", xticks=False)
  disp_matrix(Q, "Query Matrix", "Query", "i")
  disp_matrix(K, "Key Matrix", "Key", "i")
  disp_matrix(V, "Value Matrix", "Value", "i")
  
  output = attention(Q, K, V)
  
  disp_matrix(output, "Output of attention", "Output", "i", xticks=False)
  
  E_updated = E + output
  
  disp_matrix(E_updated, "Updated Embedding Matrix", "Updated Embedding", "i", xticks=False)
  
  
  
  

if __name__ == "__main__":
  main()
  