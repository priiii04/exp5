
Experiment 5 — K-Means Clustering and Inertia (Unsupervised Learning)


AIM:

To apply K-Means clustering to a synthetic 2-cluster dataset, and compute and visualize inertia.

ALGORITHM:

1. Start and import numpy, matplotlib.pyplot, and KMeans.
2. Generate synthetic data: Set random seed (42) and create two 2D Gaussian clusters centered at [0, 0] and [5, 5] with 50 points each, stacking them into dataset X.
3. Initialize and fit K-Means: Run KMeans(n_clusters=2, random_state=42).fit(X), which applies Lloyd’s algorithm (assigning points to the nearest centroid and recomputing centroids iteratively until convergence).
4. Extract results: Retrieve cluster assignments (kmeans.labels_) and final centroid coordinates (kmeans.cluster_centers_).
5. Visualize clusters: Plot the data points colored by cluster labels and overlay the learned centroids as red 'x' markers.
6. Compute inertia: Print the within-cluster sum-of-squares (kmeans.inertia_) rounded to 3 decimal places.
7. Stop.

THEORY:

K-Means is an unsupervised partitioning algorithm that groups unlabeled data into \(K\) distinct clusters by minimizing inertia (Within-Cluster Sum of Squares, or WCSS), which measures how tightly grouped data points are around their assigned centroids. Using Lloyd's algorithm, it alternates between assigning points to the nearest centroid and updating centroids to the mean of assigned points—a process closely related to vector quantization. Because it relies on Euclidean distance and random centroid initialization, K-Means is sensitive to feature scaling and initial centroid placement.

PROGRAM:

python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 2) + np.array([0, 0]),
    np.random.randn(50, 2) + np.array([5, 5])
])

kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='red', s=100)
plt.title('KMeans Clusters')
plt.show()

print('Inertia (WCSS):', round(kmeans.inertia_, 3))
Use code with caution.

DETAILED EXPLANATION:

1. Generate synthetic dataset: Two 50-point 2D Gaussian blobs are created around [0, 0] and [5, 5] and stacked vertically into a 100-sample array X.
2. Fit K-Means model: KMeans(n_clusters=2, random_state=42).fit(X) executes clustering iterations, storing cluster indices in labels_ and final centroid coordinates in cluster_centers_.
3. Visualize clusters and centroids: plt.scatter plots all data points colored by their assigned cluster (c=labels) and marks the two learned centroids with red 'x' symbols.
4. Calculate inertia: kmeans.inertia_ computes the total squared distance of all samples to their closest cluster center (WCSS).

SAMPLE OUTPUT:

text
(Scatter plot showing two clusters with red 'x' centroids)
Inertia (WCSS): 180.342
Use code with caution.

RESULT:

Thus, K-Means clustering was successfully applied to a synthetic 2-cluster dataset, demonstrating cluster separation, centroid positioning, and within-cluster sum-of-squares (inertia) evaluation while noting sensitivity to initialization and feature scaling.
