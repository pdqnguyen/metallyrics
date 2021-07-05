import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import lyrics_utils as utils


OUTPUT = 'output/graphs/genres'
DATA_FILENAME = 'bands-1pct.csv'


print('loading data')
df = utils.load_bands(DATA_FILENAME)
df = df[df['reviews'] > 50].reset_index(drop=True)
print(len(df))
names = df['name'].values

genre_cols = [c for c in df.columns if 'genre_' in c]
Y = df[genre_cols].values
n_clusters = 11
kmeans = KMeans(n_clusters=n_clusters).fit(Y)
labels = kmeans.labels_

print('saving nodes')
if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)
nodes = pd.DataFrame({'id': range(1, len(names) + 1), 'label': names, 'class': labels})
nodes_filepath = os.path.join(OUTPUT, DATA_FILENAME).replace('.csv', '-sim-nodes.csv')
nodes.to_csv(nodes_filepath, index=False)
print('saving edges')
edges_dict = {'Source': [], 'Target': []}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        if np.dot(Y[i], Y[j]) > 0:
            edges_dict['Source'].append(i + 1)
            edges_dict['Target'].append(j + 1)
edges = pd.DataFrame(edges_dict)
print(len(edges))
edges_filepath = nodes_filepath.replace('nodes', 'edges')
edges.to_csv(edges_filepath, index=False)

# name = 'Cannibal Corpse'
# print(sims_df[name].sort_values())
# sims_df[name].sort_values()[:-1].hist(bins=20)
# plt.show()
