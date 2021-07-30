from argparse import ArgumentParser
import yaml
import pandas as pd
import numpy as np


def get_config(required=('input', 'output')):
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    for key in required:
        if key not in cfg.keys():
            raise KeyError(f"missing field {key} in {args.config}")
    return cfg


def load_songs(filepath):
    data = pd.read_csv(filepath)
    data['song_words'] = data['song_words'].str.split(' ')
    return data


def load_bands(filepath):
    data = pd.read_csv(filepath)
    data['words'] = data['words'].str.split(' ')
    return data


def get_song_stats(data):
    data['words_uniq'] = data['song_words'].apply(set)
    data['word_count'] = data['song_words'].apply(len)
    data['word_count_uniq'] = data['words_uniq'].apply(len)
    data['word_rate'] = data['word_count'] / data['seconds']
    data['word_rate_uniq'] = data['word_count_uniq'] / data['seconds']
    data.loc[data['word_rate'] == np.inf, 'word_rate'] = 0
    data.loc[data['word_rate_uniq'] == np.inf, 'word_rate_uniq'] = 0
    return data


def get_band_stats(data):
    data['words_uniq'] = data['words'].apply(set)
    data['word_count'] = data['words'].apply(len)
    data['word_count_uniq'] = data['words_uniq'].apply(len)
    data['words_per_song'] = data['word_count'] / data['songs']
    data['words_per_song_uniq'] = data['word_count_uniq'] / data['songs']
    data['seconds_per_song'] = data['seconds'] / data['songs']
    data['word_rate'] = data['word_count'] / data['seconds']
    data['word_rate_uniq'] = data['word_count_uniq'] / data['seconds']
    data.loc[data['word_rate'] == np.inf, 'word_rate'] = 0
    data.loc[data['word_rate_uniq'] == np.inf, 'word_rate_uniq'] = 0
    return data


def uniq_first_words(x, num_words):
    """Of the first `num_words` in this text, how many are unique?
    """
    return len(set(x[:num_words]))


def get_band_words(data, num_bands=None, num_words=None):
    """Filter bands by word count and reviews, and count number of unique first words.
    """
    data_short = data[data['word_count'] > num_words].copy()
    top_reviewed_bands = data_short.sort_values('reviews')['name'][-num_bands:]
    data_short = data_short.loc[top_reviewed_bands.index]
    data_short['unique_first_words'] = data_short['words'].apply(uniq_first_words, args=(num_words,))
    data_short = data_short.sort_values('unique_first_words')
    return data_short


def convert_seconds(series):
    """Convert a series of time strings (MM:ss or HH:MM:ss) to seconds
    """
    out = pd.Series(index=series.index, dtype=int)
    for i, x in series.items():
        if isinstance(x, str):
            xs = x.split(':')
            if len(xs) < 3:
                xs = ['00'] + xs
            seconds = int(xs[0]) * 3600 + int(xs[1]) * 60 + int(xs[2])
        else:
            seconds = 0
        out[i] = seconds
    return out


def songs2bands(data):
    genre_cols = [c for c in data.columns if 'genre_' in c]
    out = pd.concat(
        (
            data.groupby('band_id')[['band_name', 'band_genre']].first(),
            data.groupby(['band_id', 'album_name'])['album_review_num'].first().groupby('band_id').sum(),
            data.groupby(['band_id', 'album_name'])['album_review_avg'].first().groupby('band_id').mean(),
            data.groupby('band_id').apply(len),
            data.groupby('band_id')[['song_darklyrics', 'song_words']].sum(),
            data.groupby('band_id')['seconds'].sum(),
            data.groupby('band_id')[genre_cols].first(),
        ),
        axis=1
    ).reset_index()
    out.columns = [
        'id',
        'name',
        'genre',
        'reviews',
        'rating',
        'songs',
        'lyrics',
        'words',
        'seconds',
    ] + genre_cols
    return out
