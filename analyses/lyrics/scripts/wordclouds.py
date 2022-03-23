import os
import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

import lyrics_utils as utils


def tokenizer(s):
    t = RegexpTokenizer('[a-zA-Z]+')
    return [word.lower() for word in t.tokenize(s) if len(word) >= 4]


cfg = utils.get_config(required=('input',))
output = cfg['output']
if not os.path.exists(output):
    os.mkdir(output)
band_df = utils.load_bands(cfg['input'])
band_df = utils.get_band_stats(band_df)
band_df = utils.get_band_words(band_df, num_bands=cfg['num_bands'], num_words=cfg['num_words'])
genre_cols = [c for c in band_df.columns if 'genre_' in c]
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
corpus = band_df.words.apply(' '.join).values
X = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names()
output_bands = pd.DataFrame(X.toarray(), index=band_df.name, columns=vocab)
output_bands.to_csv(os.path.join(cfg['output'], cfg['filename']))
