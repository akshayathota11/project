import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

# Load data
df = pd.read_csv("railway_cleaned.csv")

# Create binary target (optional if used elsewhere)
df['Is_Delayed'] = df['Journey Status'].apply(lambda x: 1 if x == 'Delayed' else 0)

# Convert datetime
df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], errors='coerce')
df['Departure Time'] = pd.to_datetime(df['Departure Time'], format='%H:%M:%S', errors='coerce')
df['Hour'] = df['Departure Time'].dt.hour
df['Day'] = df['Date of Journey'].dt.day_name()

# Feature selection
features = ['Departure Station', 'Arrival Destination', 'Ticket Class', 'Ticket Type', 'Railcard',
            'Purchase Type', 'Payment Method', 'Hour', 'Day', 'Price']
X = pd.get_dummies(df[features])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Silhouette scores
sil_vals = silhouette_samples(X_scaled, df['Cluster'])
avg_sil_score = silhouette_score(X_scaled, df['Cluster'])

# Silhouette plot
plt.figure(figsize=(10, 5))
y_lower = 10
for i in range(10):
    ith_cluster_sil_vals = sil_vals[df['Cluster'] == i]
    ith_cluster_sil_vals.sort()
    size_cluster_i = ith_cluster_sil_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    color = plt.cm.tab10(i / 10)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_vals, facecolor=color, edgecolor=color, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

plt.axvline(avg_sil_score, color="red", linestyle="--", label=f"Average Score = {avg_sil_score:.2f}")
plt.title("Silhouette Plot for KMeans Clustering")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.legend()
plt.tight_layout()
plt.savefig("silhouette_plot.png")
plt.show()

