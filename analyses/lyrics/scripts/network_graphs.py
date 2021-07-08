import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

import lyrics_utils as utils


OUTPUT = 'output/graphs/'
DATA_FILENAME = 'bands-1pct.csv'


print('loading data')
df = utils.load_bands(DATA_FILENAME)
df = df[df['reviews'] > 50].reset_index(drop=True)
print(len(df))
names = df['name'].values


# LYRICAL CLUSTERING
vectorizer = TfidfVectorizer(min_df=0.2, max_df=0.8)
corpus = list(df['words'].apply(lambda x: ' '.join(x)))
X = vectorizer.fit_transform(corpus)
X_svd = TruncatedSVD(n_components=2).fit_transform(X)
n_clusters = 5
lyrics_kmeans = KMeans(n_clusters=n_clusters).fit(X_svd)
lyrics_labels = lyrics_kmeans.labels_


# GENRE CLUSTERING
genre_cols = [c for c in df.columns if 'genre_' in c]
Y = df[genre_cols].values
n_clusters = 11
genre_kmeans = KMeans(n_clusters=n_clusters).fit(Y)
genre_labels = genre_kmeans.labels_

print('saving nodes')
if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)
nodes = pd.DataFrame({'id': range(1, len(names) + 1), 'label': names, 'genre': genre_labels, 'lyrics': lyrics_labels})
nodes_filepath = os.path.join(OUTPUT, DATA_FILENAME).replace('.csv', '-nodes.csv')
nodes.to_csv(nodes_filepath, index=False)


# VOCABULARY SIMILARILITY
n = len(df)
m = len(vectorizer.get_feature_names())
sims = np.zeros((n, n))
print('computing similiarities')
for i in range(n):
    if i % 10 == 0:
        print(f"\r{i}/{n}", end="")
    a = X[i, :].toarray()[0]
    for j in range(i + 1, n):
        b = X[j, :].toarray()[0]
        sims[i, j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
sims += np.rot90(np.fliplr(sims))
np.fill_diagonal(sims, 1.0)
sims_df = pd.DataFrame(sims, index=df['name'], columns=df['name'])

print('saving vocabulary-similarity edges')
edges_dict = {'Source': [], 'Target': []}
for i in range(len(names)):
    row = sims_df.iloc[i, i+1:]
    if len(row) > 0:
        thresh = np.percentile(row.values, 95)
        for j in range(len(row)):
            if row[j] > thresh:
                edges_dict['Source'].append(i + 1)
                edges_dict['Target'].append(i + j + 2)
edges = pd.DataFrame(edges_dict)
print(len(edges))
edges_filepath = nodes_filepath.replace('nodes', 'vocab-edges')
edges.to_csv(edges_filepath, index=False)

# GENRE ASSOCIATION
print('saving genre-association edges')
edges_dict = {'Source': [], 'Target': []}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        if np.dot(Y[i], Y[j]) > 0:
            edges_dict['Source'].append(i + 1)
            edges_dict['Target'].append(j + 1)
edges = pd.DataFrame(edges_dict)
print(len(edges))
edges_filepath = nodes_filepath.replace('nodes', 'genre-edges')
edges.to_csv(edges_filepath, index=False)
