# Importing Required Libraries
# I installed the necessary libraries.
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Loading the Dataset
# Load the Wholesale Customers Data dataset.
# This dataset includes features like milk, grocery, frozen etc.
data = pd.read_csv("data/wholesale_customers_data.csv")

# If 'Channel' and 'Region' columns there are, then drop them.
# Because they are not necessary for clustering.
if 'Channel' in data.columns and 'Region' in data.columns:
    data.drop(['Channel', 'Region'], axis=1, inplace=True)

# 2. Data Preprocessing
# I created a scaler for scaling the features.
scaler = StandardScaler()
# scaled data feature
scaled_data = scaler.fit_transform(data)

# 3. Determining the Number of Clusters
# Calculate variance reduction ratio for the Elbow method. It will help to find k value.
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=58)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Calculate variance reduction ratio.
variance_ratio = [1 - (inertia[i] / inertia[0]) for i in range(len(inertia))]

# Drawing Elbow chart.
plt.figure(figsize=(8, 5))
plt.plot(k_range, variance_ratio, marker='o')
plt.title("Determining the Number of Clusters with the Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Variance Reduction Ratio")
plt.show()

# 4. K-Means Clustering
# Selecting the optimal number of clusters based on the Elbow method.
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=58)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster results to the dataset.
data['Cluster'] = clusters

# 5. Determining the Clustering Quality
# Silhouette Score calculation. It will show how well we separate the clusters.

silhouette_avg = silhouette_score(scaled_data, clusters)
print(f"\nSilhouette Score for k={optimal_k}: {silhouette_avg}")

# 6. Analysis and Visualization of Results
# Scatter plot for two variables (e.g., first two dimensions of scaled data)
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=scaled_data[:, 0],
    y=scaled_data[:, 1],
    hue=data['Cluster'],
    palette='viridis',
    s=50
)
plt.title("K-Means Clustering Results")
plt.xlabel("Scaled Feature 1")
plt.ylabel("Scaled Feature 2")
plt.legend(title="Clusters")
plt.show()