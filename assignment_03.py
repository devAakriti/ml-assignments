# Dataset: built into scikit-learn (Wine dataset, 178 samples, 13 features)
# Also available at https://archive.ics.uci.edu/dataset/109/wine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 1. Load data and standardise
data = load_wine()
X, y = data.data, data.target
print("Original shape:", X.shape)
X_std = StandardScaler().fit_transform(X)

# 2. PCA from scratch using eigen decomposition
cov = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eigh(cov)
order = np.argsort(eig_vals)[::-1]
eig_vals, eig_vecs = eig_vals[order], eig_vecs[:, order]
print("Explained variance ratio (manual):")
print(np.round(eig_vals / eig_vals.sum(), 4))

# 3. PCA using scikit-learn
pca = PCA().fit(X_std)
ratio = pca.explained_variance_ratio_
cum = np.cumsum(ratio)
print("Cumulative variance:", np.round(cum, 4))
n95 = np.argmax(cum >= 0.95) + 1
print("Components needed for 95% variance:", n95)

# 4. Scree plot and cumulative variance plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(range(1, 14), ratio, color="gray", edgecolor="black")
ax[0].set_title("Scree Plot")
ax[0].set_xlabel("Principal component")
ax[0].set_ylabel("Explained variance ratio")
ax[1].plot(range(1, 14), cum, "ko-")
ax[1].axhline(0.95, color="black", linestyle="--")
ax[1].set_title("Cumulative Explained Variance")
ax[1].set_xlabel("Number of components")
plt.show()

# 5. Reduce to 2 dimensions and visualise
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_std)
plt.figure(figsize=(7, 5))
for cls, marker in zip(range(3), ["o", "s", "^"]):
    plt.scatter(X_pca[y == cls, 0], X_pca[y == cls, 1], marker=marker,
                label=data.target_names[cls])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Wine dataset projected on 2 principal components")
plt.legend()
plt.show()

# 6. Feature loadings of the first two components
loadings = pd.DataFrame(pca2.components_.T, index=data.feature_names,
                        columns=["PC1", "PC2"])
print(loadings.round(3))

# 7. Effect of PCA on classification accuracy
clf = LogisticRegression(max_iter=1000)
acc_full = cross_val_score(clf, X_std, y, cv=5).mean()
X_n95 = PCA(n_components=n95).fit_transform(X_std)
acc_pca = cross_val_score(clf, X_n95, y, cv=5).mean()
print("Accuracy with 13 features :", round(acc_full, 4))
print(f"Accuracy with {n95} components:", round(acc_pca, 4))
