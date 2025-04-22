import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Example data: 2D array or pandas DataFrame
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Optional: set a style
sns.set()

# Create heatmap
sns.heatmap(data,annot=True, cmap="YlGnBu")

# Show the plot
plt.title("Attention pattern between the dot product of every pair of key and query vectors")
plt.xlabel("Query vectors")
plt.ylabel("Key vectors")
plt.show()