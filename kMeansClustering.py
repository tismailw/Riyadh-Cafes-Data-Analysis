#Using K-Means Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv(r"C:\Users\tahir\OneDrive\Desktop\Python\cds\riyadh_cafes.csv")


subset_data = data[['rating_count', '24_hours']].copy()

def map_24_hours(x):
    if x == 'TRUE':
        return 1
    else:
        return 0

subset_data['24_hours'] = subset_data['24_hours'].apply(map_24_hours)
subset_data.dropna(inplace = True)


k = 2
kmeans = KMeans(n_clusters=k)
kmeans.fit(subset_data)
cluster_assignments = kmeans.predict(subset_data)

for i in range(k):
    cluster_points = subset_data[cluster_assignments == i]
    print(f"Cluster {i+1}:")
    print(cluster_points)

colors = ['r', 'g']
for i in range(k):
    cluster_points = subset_data[cluster_assignments == i]
    plt.scatter(cluster_points['rating_count'], cluster_points['24_hours'], color=colors[i])

plt.xlabel("rating_count")
plt.ylabel("24_hours")
plt.title("K-means Clustering: rating_count vs. 24_hours")
plt.show()
