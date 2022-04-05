import os
import numpy as np
import pandas as pd

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import lyrics_utils as utils


def tokenizer(s):
    return [word for word in s.split() if len(word) >= 4]


def clean_words(words, minlen=4):
    stop_words = stopwords.words('english')
    words = [w for w in words if (w not in stop_words) and (len(w) >= minlen)]
    return words


print('loading data')
cfg = utils.get_config(required=('input',))
df = utils.load_bands(cfg['input'])
df = utils.get_band_stats(df)
df_short = utils.get_band_words(df, num_bands=cfg['num_bands'], num_words=cfg['num_words'])
names_short = df_short['name'].values
names = df['name'].values
df['clean_words'] = df['words'].apply(clean_words)

# VOCABULARY CLUSTERING
vectorizer = TfidfVectorizer(
    stop_words=stopwords.words('english'),
    tokenizer=tokenizer,
    min_df=cfg['min_df'],
    max_df=cfg['max_df'],
    max_features=cfg['max_features'],
    sublinear_tf=cfg['sublinear_tf'],
)
corpus = list(df['clean_words'].apply(lambda x: ' '.join(x)))
X = vectorizer.fit_transform(corpus)
X_svd = TruncatedSVD(n_components=2, random_state=0).fit_transform(X)
vocab_kmeans = KMeans(n_clusters=cfg['vocab_n_clusters'], random_state=0).fit(X_svd)
vocab_labels = vocab_kmeans.labels_
print("Lyrics clustering results:")
for i in sorted(set(vocab_labels)):
    print(f"{i}: {sum(vocab_labels == i) / len(vocab_labels) * 100:.1f}%")

# GENRE CLUSTERING
genre_cols = [c for c in df.columns if 'genre_' in c]
Y = df[genre_cols].values
Y_pca = PCA(n_components=2, random_state=0).fit_transform(Y)
genre_kmeans = KMeans(n_clusters=cfg['genre_n_clusters'], random_state=0).fit(Y_pca)
genre_labels = genre_kmeans.labels_
print("\n\nGenre clustering results:")
for i in sorted(set(genre_labels)):
    print(f"{i}: {sum(genre_labels == i) / len(genre_labels) * 100:.1f}%")
for i in sorted(set(genre_labels)):
    center_genres = sorted(zip(genre_cols, Y[genre_labels == i].mean(axis=0)), key=lambda x: -x[1])
    print(", ".join([f"{genre.split('_')[1]} {value:.2f}" for genre, value in center_genres if value > 0.1]))

# Combine vocab and genre cluster labels and save as a table of nodes
print('\n\nsaving nodes')
if not os.path.exists(cfg['output']):
    os.mkdir(cfg['output'])
nodes = pd.DataFrame({'id': range(1, len(names) + 1), 'label': names, 'genre': genre_labels, 'lyrics': vocab_labels})
nodes = nodes[nodes.label.isin(names_short)].reset_index(drop=True)
nodes_filepath = os.path.join(cfg['output'], os.path.basename(cfg['input'])).replace('.csv', '-nodes.csv')
nodes.to_csv(nodes_filepath, index=False)

# VOCABULARY SIMILARILITY
print('saving vocabulary edges')
cos_matrix = cosine_similarity(X)
cos_vals = cos_matrix[cos_matrix < 1]
vocab_edges_dict = {'Source': [], 'Target': []}
for i, name in enumerate(names):
    row = cos_matrix[i, :]
    vals = row[row < 1]
    thresh = vals.mean() + cfg['vocab_cosine_thresh'] * vals.std()
    idx = np.where(row > thresh)[0]
    idx = idx[idx != i]
    for j in idx:
        vocab_edges_dict['Source'].append(i + 1)
        vocab_edges_dict['Target'].append(j + 1)
vocab_edges = pd.DataFrame(vocab_edges_dict)
vocab_edges = vocab_edges[vocab_edges.Source.isin(nodes.id) & vocab_edges.Target.isin(nodes.id)]
vocab_edges_filepath = nodes_filepath.replace('nodes', 'vocab-edges')
vocab_edges.to_csv(vocab_edges_filepath, index=False)

# GENRE SIMILARILITY
print('saving genre-association edges')
genre_edges_dict = {'Source': [], 'Target': []}
for i, row1 in nodes.iterrows():
    id_i = row1.id
    for j, row2 in nodes[nodes.id > id_i].iterrows():
        id_j = row2.id
        if np.dot(Y[id_i - 1], Y[id_j - 1]) > 0:
            genre_edges_dict['Source'].append(id_i)
            genre_edges_dict['Target'].append(id_j)
genre_edges = pd.DataFrame(genre_edges_dict)
genre_edges_filepath = nodes_filepath.replace('nodes', 'genre-edges')
genre_edges.to_csv(genre_edges_filepath, index=False)
