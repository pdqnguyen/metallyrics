from argparse import ArgumentParser
import yaml
import pandas as pd


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
