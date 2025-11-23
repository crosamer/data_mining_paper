import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform

# Load Dataset
df = pd.read_csv("D:/College/Semester_5/Data Mining/coding/tugas paper/clustering/dataset500.csv", low_memory=False)

# Attribut untuk Klasterisasi
selected_features = [
    "Work_Hours_Per_Week", 
    "Sick_Days",
    "Overtime_Hours",
    "Performance_Score",
    "Remote_Work_Frequency",
    "Employee_Satisfaction_Score"
]

data = df[selected_features]

# Tangani Missing Values
data = data.replace("?", np.nan)
data = data.astype(float)
data = data.fillna(data.mean())

# Normalisasi Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(scaled_data)

# Hierarchical Clustering
hier = AgglomerativeClustering(n_clusters=3)
df["Hierarchical_Cluster"] = hier.fit_predict(scaled_data)

# Fungsi Dunn Index
def dunn_index(X, labels):
    distances = squareform(pdist(X))
    unique_clusters = np.unique(labels)

    intra_cluster = []
    inter_cluster = []

    for cluster in unique_clusters:
        cluster_points = np.where(labels == cluster)[0]
        if len(cluster_points) > 1:
            intra = np.max(distances[np.ix_(cluster_points, cluster_points)])
            intra_cluster.append(intra)
        else:
            intra_cluster.append(0)

    for i in unique_clusters:
        for j in unique_clusters:
            if i < j:
                points_i = np.where(labels == i)[0]
                points_j = np.where(labels == j)[0]
                inter = np.min(distances[np.ix_(points_i, points_j)])
                inter_cluster.append(inter)

    return np.min(inter_cluster) / np.max(intra_cluster)

# Evaluasi K-Means & Hierarchical
sil_kmeans = silhouette_score(scaled_data, df["KMeans_Cluster"])
ch_kmeans = calinski_harabasz_score(scaled_data, df["KMeans_Cluster"])
dunn_kmeans = dunn_index(scaled_data, df["KMeans_Cluster"])

sil_hier = silhouette_score(scaled_data, df["Hierarchical_Cluster"])
ch_hier = calinski_harabasz_score(scaled_data, df["Hierarchical_Cluster"])
dunn_hier = dunn_index(scaled_data, df["Hierarchical_Cluster"])

print("\n======== Evaluasi K-Means =========")
print("Silhouette Score :", sil_kmeans)
print("Calinski-Harabasz Index :", ch_kmeans)
print("Dunn Index :", dunn_kmeans)

print("\n======== Evaluasi Hierarchical =========")
print("Silhouette Score :", sil_hier)
print("Calinski-Harabasz Index :", ch_hier)
print("Dunn Index :", dunn_hier)

# Visualisasi 3D K-Means
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    df["Work_Hours_Per_Week"],
    df["Performance_Score"],
    df["Employee_Satisfaction_Score"],
    c=df["KMeans_Cluster"],
    s=60
)

ax.set_xlabel("Work_Hours_Per_Week")
ax.set_ylabel("Performance_Score")
ax.set_zlabel("Employee_Satisfaction_Score")
ax.set_title("Visualisasi 3D Klaster K-Means")
plt.show()


# Visualisasi 3D Hierarchical
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    df["Work_Hours_Per_Week"],
    df["Performance_Score"],
    df["Employee_Satisfaction_Score"],
    c=df["Hierarchical_Cluster"],
    s=60
)

ax.set_xlabel("Work_Hours_Per_Week")
ax.set_ylabel("Performance_Score")
ax.set_zlabel("Employee_Satisfaction_Score")
ax.set_title("Visualisasi 3D Klaster Hierarchical")
plt.show()
