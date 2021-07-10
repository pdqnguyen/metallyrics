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
    sublinear_tf=cfg['sublinear_tf'],
)
corpus = []
for genre, col in zip(genres, genre_cols):
    other_cols = [c for c in genre_cols if c != col]
    words = df[(df[col] == 1) & (df[other_cols] == 0).all(axis=1)].words
    corpus.append(' '.join(words))
X = vectorizer.fit_transform(corpus)
vocabulary = vectorizer.get_feature_names()

for i, genre in enumerate(genres):
    print(f"producing wordcloud for genre: {genre}")
    freqs = X.toarray()[i, :]
    word_freqs = dict(zip(vocabulary, freqs))
    word_cloud = WordCloud(width=800, height=500).fit_words(word_freqs)
    plt.figure(figsize=(8, 5))
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.savefig(os.path.join(cfg['output'], genre + '.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

# texts = []
# print("tokenizing corpus")
# for genre in genres:
#     song_tokens = df[df[genre] == 1].lyrics.apply(lambda x: ' '.join(tokenize(x, **tokenize_kwargs)))
#     genre_tokens = ' '.join(song_tokens).split()
#     texts.append(genre_tokens)

# print("producing corpus wordcloud")
# full_text = sum(texts, [])
# word_freqs = get_wordcloud_frequencies([full_text])
# word_cloud = WordCloud(width=800, height=500).fit_words(word_freqs[0])
# plt.figure(figsize=(8, 5))
# plt.imshow(word_cloud)
# plt.axis('off')
# plt.savefig(os.path.join(cfg['output'], 'full.png'), bbox_inches='tight', pad_inches=0)
# plt.close()

# word_freqs = get_wordcloud_frequencies(texts, tfidf=True, min_tf_pct=0.01)
# for i, genre in enumerate(genres):
#     print(f"producing genre wordcloud: {genre}")
#     word_cloud = WordCloud(width=800, height=500).fit_words(word_freqs[i])
#     plt.figure(figsize=(8, 5))
#     plt.imshow(word_cloud)
#     plt.title(f'Genre: {genre}', fontsize=14)
#     plt.axis('off')
#     plt.savefig(os.path.join(subdir, genre + '.png'), bbox_inches='tight', pad_inches=0)
