import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

import lyrics_utils as utils


OUTPUT = 'output/graphs/lyrics'
DATA_FILENAME = 'bands-1pct.csv'


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print('loading data')
df = utils.load_bands(DATA_FILENAME)
df = df[df['reviews'] > 50].reset_index(drop=True)
print(len(df))
names = df['name'].values

vectorizer = TfidfVectorizer(min_df=0.2, max_df=0.8)
corpus = list(df['words'].apply(lambda x: ' '.join(x)))
X = vectorizer.fit_transform(corpus)
Y = df[[c for c in df.columns if 'genre_' in c]].values

X_svd = TruncatedSVD(n_components=2).fit_transform(X)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters).fit(X_svd)
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

# Y = df[[c for c in df.columns if 'genre_' in c]].values
# n_clusters = 6
# kmeans = KMeans(n_clusters=n_clusters).fit(Y)
# labels = kmeans.labels_

# n = len(df)
# m = len(vectorizer.get_feature_names())
# sims = np.zeros((n, n))
# print('computing similiarities')
# for i in range(n):
#     if i % 10 == 0:
#         print(f"\r{i}/{n}", end="")
#     text1 = X[i, :].toarray()[0]
#     for j in range(i + 1, n):
#         text2 = X[j, :].toarray()[0]
#         sims[i, j] = cos_sim(text1, text2)
# sims += np.rot90(np.fliplr(sims))
# np.fill_diagonal(sims, 1.0)
#
# print('\nsaving similiarity matrix')
# sims_df = pd.DataFrame(sims, index=df['name'], columns=df['name'])
# sims_filepath = os.path.join(OUTPUT, DATA_FILENAME.replace('.csv', '-sim.csv'))
# sims_df.to_csv(sims_filepath, index=False)
# names = list(sims_df.columns)
#
# print('saving nodes')
# if not os.path.exists(OUTPUT):
#     os.mkdir(OUTPUT)
# nodes = pd.DataFrame({'id': range(1, len(names) + 1), 'label': names, 'class': labels})
# nodes_filepath = sims_filepath.replace('.csv', '-nodes.csv')
# nodes.to_csv(nodes_filepath, index=False)
# print('saving edges')
# edges_dict = {'Source': [], 'Target': []}
# for i in range(len(names)):
#     row = sims_df.iloc[i, i+1:]
#     if len(row) > 0:
#         thresh = np.percentile(row.values, 95)
#         for j in range(len(row)):
#             if row[j] > thresh:
#                 edges_dict['Source'].append(i + 1)
#                 edges_dict['Target'].append(i + j + 2)
# edges = pd.DataFrame(edges_dict)
# print(len(edges))
# edges_filepath = nodes_filepath.replace('nodes', 'edges')
# edges.to_csv(edges_filepath, index=False)

# name = 'Cannibal Corpse'
# print(sims_df[name].sort_values())
# sims_df[name].sort_values()[:-1].hist(bins=20)
# plt.show()
