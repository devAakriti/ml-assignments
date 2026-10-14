# Dataset: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
# Download Mall_Customers.csv and keep it in the working directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load data and select the features used for clustering
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
X_scaled = StandardScaler().fit_transform(X)

# 2. Elbow method and silhouette score to choose K
wcss, sil = [], []
K = range(2, 11)
for k in K:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    wcss.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(K, wcss, "ko-")
ax[0].set_title("Elbow Method")
ax[0].set_xlabel("Number of clusters K")
ax[0].set_ylabel("WCSS")
ax[1].plot(K, sil, "ko-")
ax[1].set_title("Silhouette Score")
ax[1].set_xlabel("Number of clusters K")
ax[1].set_ylabel("Score")
plt.show()

# 3. Final K-Means model (K = 5)
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)
print("Silhouette score:", silhouette_score(X_scaled, df["KMeans_Cluster"]))

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=df["KMeans_Cluster"], cmap="tab10", s=40)
centers = StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=200,
            label="Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clustering (K = 5)")
plt.legend()
plt.show()

# 4. Cluster profile
print(df.groupby("KMeans_Cluster")[["Age", "Annual Income (k$)",
                                    "Spending Score (1-100)"]].mean())

# 5. Hierarchical (agglomerative) clustering with Ward linkage
Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=30)
plt.title("Dendrogram (Ward linkage)")
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.show()

agg = AgglomerativeClustering(n_clusters=5, linkage="ward")
df["Hier_Cluster"] = agg.fit_predict(X_scaled)
print("Hierarchical silhouette:", silhouette_score(X_scaled, df["Hier_Cluster"]))
print(df["Hier_Cluster"].value_counts())
