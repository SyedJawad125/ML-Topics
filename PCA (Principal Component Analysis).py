from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Create fake data: 100 samples, 4 features
np.random.seed(42)
X = np.random.rand(100, 4)

# Apply PCA to reduce from 4D to 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)

# Plot reduced data
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='blue')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Reduced Data")
plt.show()
