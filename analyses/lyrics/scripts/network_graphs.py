import os
import numpy as np
import pandas as pd

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import DBSCAN
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
df = utils.get_band_words(df, num_bands=cfg['num_bands'], num_words=cfg['num_words'])
names = df['name'].values
df['clean_words'] = df['words'].apply(clean_words)


# LYRICAL CLUSTERING
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
vocab_dbscan = DBSCAN(eps=cfg['vocab_eps'], min_samples=cfg['vocab_min_samples']).fit(X_svd)
vocab_labels = vocab_dbscan.labels_
print("Lyrics clustering results:")
for i in sorted(set(vocab_labels)):
    print(f"{i}: {sum(vocab_labels == i) / len(vocab_labels) * 100:.1f}%")


# GENRE CLUSTERING
genre_cols = [c for c in df.columns if 'genre_' in c]
Y = df[genre_cols].values
Y_pca = PCA(n_components=2, random_state=0).fit_transform(Y)
genre_dbscan = DBSCAN(eps=cfg['genre_eps'], min_samples=cfg['genre_min_samples']).fit(Y_pca)
genre_labels = genre_dbscan.labels_
print("\n\nGenre clustering results:")
for i in sorted(set(genre_labels)):
    print(f"{i}: {sum(genre_labels == i) / len(genre_labels) * 100:.1f}%")
for i in sorted(set(genre_labels)):
    center_genres = sorted(zip(genre_cols, Y[genre_labels == i].mean(axis=0)), key=lambda x: -x[1])
    print(", ".join([f"{genre.split('_')[1]} {value:.2f}" for genre, value in center_genres if value > 0.1]))


print('\n\nsaving nodes')
if not os.path.exists(cfg['output']):
    os.mkdir(cfg['output'])
nodes = pd.DataFrame({'id': range(1, len(names) + 1), 'label': names, 'genre': genre_labels, 'lyrics': vocab_labels})
nodes_filepath = os.path.join(cfg['output'], os.path.basename(cfg['input'])).replace('.csv', '-nodes.csv')
nodes.to_csv(nodes_filepath, index=False)


# VOCABULARY SIMILARILITY
edges_per_node = 5
cosine_matrix = cosine_similarity(X)
edges_dict = {'Source': [], 'Target': []}
for i, name in enumerate(names):
    ranking = np.argsort(cosine_matrix[i, :])[::-1]
    ranking = ranking[ranking != i]
    for j in ranking[:edges_per_node]:
        edges_dict['Source'].append(i + 1)
        edges_dict['Target'].append(j + 1)

edges = pd.DataFrame(edges_dict)
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
edges_filepath = nodes_filepath.replace('nodes', 'genre-edges')
edges.to_csv(edges_filepath, index=False)
