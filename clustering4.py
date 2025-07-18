import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv("railway_cleaned.csv")

# Add binary delay column
df['Is_Delayed'] = df['Journey Status'].apply(lambda x: 1 if x == 'Delayed' else 0)

# Feature Engineering
df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], errors='coerce')
df['Departure Time'] = pd.to_datetime(df['Departure Time'], format='%H:%M:%S', errors='coerce')
df['Hour'] = df['Departure Time'].dt.hour
df['Day'] = df['Date of Journey'].dt.day_name()

# Select features for clustering
features = ['Departure Station', 'Arrival Destination', 'Ticket Class', 'Ticket Type',
            'Railcard', 'Purchase Type', 'Payment Method', 'Hour', 'Day']

df_cluster = pd.get_dummies(df[features])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cluster)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=2)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Cluster sizes
print("\n Cluster Distribution:\n", df['Cluster'].value_counts().rename_axis('Cluster').to_frame('Count'))

# Average delay rate per cluster
print("\n Average Delay Rate by Cluster:")
print(df.groupby('Cluster')['Is_Delayed'].mean().to_frame('Delay Rate'))

# Save Cluster Plot
plt.figure(figsize=(8, 4))
df['Cluster'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Number of Records per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Journeys")
plt.tight_layout()
plt.savefig("clusters_kmeans.png")
plt.show()
