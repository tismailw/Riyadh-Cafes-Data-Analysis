#Using tSNE Algorithim to accomplish the same goal, the reason why I tried to accomplish the same goal using this algorithim was to see if I would get the same result.

# importing the libraies I need to accomplish what I need to accomplish 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# reads the data from the CSV file
data = pd.read_csv(r"C:\Users\tahir\OneDrive\Desktop\Python\cds\riyadh_cafes.csv")

# extracts the 2 variables I want to compare from the data in the CSV file
hours_24  = data['24_hours']
rating_count = data['rating_count']

# combining the 2 variables into one dataset
data_combined = pd.concat([hours_24 , rating_count], axis=1)

# normalizing the dataset so one data point doesnt dominate the outcome.
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data_combined)

# applying t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
reduced_data = tsne.fit_transform(normalized_data)


# the graphing aspect of everything
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization')
plt.show()
