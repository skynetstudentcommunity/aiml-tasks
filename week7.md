# Week 7: Unsupervised Learning — Tutorial

## Why this week matters
Sometimes you don't have labels at all — you just want to discover
structure in data (customer segments, anomalies, compressed representations).

---

## 1. K-Means Clustering

Assigns points to K clusters by repeatedly: (1) assigning each point to its
nearest cluster center, (2) recomputing centers as the mean of assigned
points, until it stabilizes.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
```

**Choosing K:** plot the "elbow curve" (inertia vs. K) and pick the point
where adding more clusters stops helping much.

```python
inertias = [KMeans(n_clusters=k, n_init=10).fit(X_scaled).inertia_ for k in range(1, 10)]
plt.plot(range(1, 10), inertias, marker="o")
```

---

## 2. Hierarchical Clustering

Builds a tree of clusters (a dendrogram) by repeatedly merging the closest
pairs. Useful when you don't want to pick K upfront — you can cut the tree
at whatever level makes sense.

```python
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X_scaled, method="ward")
dendrogram(Z)
```

---

## 3. PCA (Dimensionality Reduction)

Compresses many correlated features into a few "principal components" that
capture most of the variance — useful for visualization and for speeding up
downstream models.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
pca.explained_variance_ratio_   # how much info each component captures
```

---

## 4. DBSCAN (optional)

Clusters based on density rather than distance to a center — good at
finding irregularly shaped clusters and flagging outliers as noise.

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X_scaled)
```

---

## Project
Cluster a customer/product dataset:
- [ ] Scale features, run K-Means, pick K via elbow method
- [ ] Reduce to 2D with PCA and visualize the clusters
- [ ] Write a short interpretation of what each cluster represents

## Resources
- StatQuest: K-means clustering — https://www.youtube.com/watch?v=4b5d3muPQmA
- StatQuest: PCA — https://www.youtube.com/watch?v=FgakZw6K1QQ
- Scikit-learn: Clustering docs — https://scikit-learn.org/stable/modules/clustering.html
