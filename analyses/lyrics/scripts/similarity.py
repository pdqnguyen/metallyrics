import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from sklearn.feature_extraction.text import TfidfVectorizer

import lyrics_utils as utils


def jaccard(a, b):
    intersection = a.intersection(b)
    union = a.union(b)
    return len(intersection) / len(union)


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_word_frequencies(texts, tfidf=False):
    print("creating dictionary")
    dictionary = Dictionary(texts)
    print("generating bag-of-words representation")
    vectors = [dictionary.doc2bow(text) for text in texts]
    if tfidf:
        print("generating TF-IDF model")
        tfidf = TfidfModel(vectors)
        vectors = [tfidf[vector] for vector in vectors]
    print("generating frequency dictionaries for WordCloud")
    out = []
    for vector in vectors:
        out.append({dictionary[id_]: count for id_, count in vector})
    return out


print('loading data')
df = utils.load_bands('bands-1pct.csv')
df = df[df['reviews'] > 50].reset_index(drop=True)
print(len(df))

# index = df.index
# corpus = list(df['words'].values)
# word_freqs = get_word_frequencies(corpus)
# df['wordset'] = [set(k for k, v in word_freqs[i].items() if v > 0.1) for i in range(len(df))]
# n = len(df)
# sims = np.zeros((n, n))
# print('computing similiarities')
# for i, x in df['wordset'].items():
#     if i % 10 == 0:
#         print(i)
#     subset = df['wordset'].iloc[index > i]
#     for j, y in subset.items():
#         sims[i, j] = jaccard(x, y)
# sims += np.rot90(np.fliplr(sims))
# np.fill_diagonal(sims, 1.0)
# sims -= sims.mean()
# sims /= (2 * sims.std())

vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.9)
corpus = list(df['words'].apply(lambda x: ' '.join(x)))
X = vectorizer.fit_transform(corpus)
n = len(df)
m = len(vectorizer.get_feature_names())
sims = np.zeros((n, n))
print('computing similiarities')
for i in range(n):
    if i % 10 == 0:
        print(i)
    a = X[i, :].toarray()[0]
    for j in range(i + 1, n):
        b = X[j, :].toarray()[0]
        sims[i, j] = cos_sim(a, b)
sims += np.rot90(np.fliplr(sims))
np.fill_diagonal(sims, 1.0)

print('saving similiarity matrix')
sims_df = pd.DataFrame(sims, index=df['name'], columns=df['name'])
sims_df.to_csv('output/bands-1pct-sim.csv', index=False)
names = list(sims_df.columns)

print('saving nodes')
nodes = pd.DataFrame({'id': range(1, len(names) + 1), 'label': names})
nodes.to_csv('output/bands-1pct-sim-nodes.csv', index=False)
print('saving edges')
edges_dict = {'Source': [], 'Target': []}#, 'Weight': []}
for i in range(len(names)):
    row = sims_df.iloc[i, i+1:]
    if len(row) > 0:
        thresh = np.percentile(row.values, 95)
        for j in range(len(row)):
            if row[j] > thresh:
                edges_dict['Source'].append(i + 1)
                edges_dict['Target'].append(i + j + 2)
                # edges_dict['Weight'].append(row[j])
edges = pd.DataFrame(edges_dict)
print(len(edges))
# edges['Weight'] -= edges['Weight'].mean()
edges.to_csv('output/bands-1pct-sim-edges.csv', index=False)

name = 'Cannibal Corpse'
print(sims_df[name].sort_values())
sims_df[name].sort_values()[:-1].hist(bins=20)
plt.show()
