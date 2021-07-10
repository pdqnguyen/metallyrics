import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords

import lyrics_utils as utils


def tokenizer(s):
    return [word for word in s.split() if len(word) >= 4]


cfg = utils.get_config(required=('input',))
df = pd.read_csv('bands-1pct.csv')
genre_cols = [c for c in df.columns if 'genre_' in c]
genres = [c.replace('genre_', '') for c in genre_cols]
if not os.path.exists(cfg['output']):
    os.makedirs(cfg['output'])

vectorizer = TfidfVectorizer(
    stop_words=stopwords.words('english'),
    tokenizer=tokenizer,
    min_df=cfg['min_df'],
    max_df=cfg['max_df'],
    max_features=cfg['max_features'],
    sublinear_tf=cfg['sublinear_tf'],
)
corpus_genres = []
for genre, col in zip(genres, genre_cols):
    other_cols = [c for c in genre_cols if c != col]
    words = df[(df[col] == 1) & (df[other_cols] == 0).all(axis=1)].words
    corpus_genres.append(' '.join(words))
X_genres = vectorizer.fit_transform(corpus_genres)
vocab_genres = vectorizer.get_feature_names()
output_genres = pd.DataFrame(X_genres.toarray(), index=genres, columns=vocab_genres)
output_genres.to_csv(os.path.join(cfg['output'], 'tfidf-genres.csv'))

for i, genre in enumerate(genres):
    print(f"producing wordcloud for genre: {genre}")
    freqs = X_genres.toarray()[i, :]
    word_freqs = dict(zip(vocab_genres, freqs))
    word_cloud = WordCloud(width=800, height=500).fit_words(word_freqs)
    plt.figure(figsize=(8, 5))
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.savefig(os.path.join(cfg['output'], genre + '.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

corpus_bands = list(df.words)
X_bands = vectorizer.fit_transform(corpus_bands)
vocab_bands = vectorizer.get_feature_names()
output_bands = pd.DataFrame(X_bands.toarray(), index=df.name, columns=vocab_bands)
output_bands.to_csv(os.path.join(cfg['output'], 'tfidf-bands.csv'))
