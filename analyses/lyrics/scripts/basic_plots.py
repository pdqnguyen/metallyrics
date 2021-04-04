"""
Plot basic properties of the data set
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

import lyrics_utils as utils
import nlp


plt.style.use('seaborn')
plt.rcParams.update(
    {
        'figure.facecolor': 'white',
        'figure.titlesize': 25,
    }
)


def songs2albums(data):
    """
    Combine songs that belong to the same albums.
    """
    album_groups = data.groupby(['band_id', 'album_name'])
    albums = pd.concat(
        (
            album_groups[['band_name', 'band_genre']].first(),
            album_groups[[c for c in data.columns if 'genre_' in c]].first(),
            album_groups[['word_count', 'word_count_uniq', 'word_rate', 'word_rate_uniq', 'seconds']].sum(),
        ),
        axis=1
    ).reset_index()
    return albums


def albums2bands(data):
    """
    Combine albums that belong to the same bands.
    """
    band_groups = data.groupby('band_id')
    bands = pd.concat(
        (
            band_groups[['band_name', 'band_genre']].first(),
            band_groups[[c for c in data.columns if 'genre_' in c]].first(),
            band_groups[['word_count', 'word_count_uniq', 'word_rate', 'word_rate_uniq', 'seconds']].sum(),
        ),
        axis=1
    ).reset_index()
    return bands


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


def plot_word_hist(data, bins, category, filepath):
    """
    Histograms of word counts and unique word counts.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Distribution of word counts of {category}")
    data['word_count'].hist(bins=bins, ax=ax1)
    data['word_count_uniq'].hist(bins=bins, ax=ax2)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax1.set_xlabel("Word count")
    ax1.set_ylabel(f"Number of {category}")
    ax2.set_xlabel("Unique word count")
    plt.savefig(filepath)


def plot_song_words(data, filepath):
    bins = np.logspace(1, 3, 30)
    plot_word_hist(data, bins, 'songs', filepath)


def plot_album_words(data, filepath):
    bins = np.logspace(2, 4, 30)
    plot_word_hist(data, bins, 'albums', filepath)


def plot_band_words(data, filepath):
    bins = np.logspace(2, 5, 30)
    plot_word_hist(data, bins, 'bands', filepath)


def plot_box(data, column, title, filepath):
    """
    Horizontal violin plot of distributions over multiple classes.
    """
    box_columns = ['genre', column]
    dtypes = {'genre': object, column: float}
    box_data = pd.DataFrame(columns=box_columns).astype(dtypes)
    for c in data.columns:
        if 'genre_' in c:
            genre_data = pd.DataFrame({
                'genre': c.replace('genre_', ''),
                column: data.loc[data[c] == 1, column].values
            })
            box_data = box_data.append(genre_data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    medians = box_data.groupby('genre')[column].median()
    order = medians.sort_values(ascending=False).index
    sns.boxplot(x=column, y='genre', data=box_data, orient='h', color='c', showfliers=False,
                order=order)
    ax.set_xlabel(title)
    ax.set_ylabel("")
    plt.savefig(filepath)


def plot_genre_words(data, filepath):
    plot_box(data, 'word_count', "Words per song", filepath)


def plot_genre_word_rate(data, filepath):
    plot_box(data, 'word_rate', "Words per second", filepath)


def plot_genre_words_uniq(data, filepath):
    plot_box(data, 'word_count_uniq', "Unique words per song", filepath)


def plot_genre_word_rate_uniq(data, filepath):
    plot_box(data, 'word_rate_uniq', "Unique words per second", filepath)


def main():
    cfg = utils.get_config()
    output = cfg['output']
    if not os.path.exists(output):
        os.mkdir(output)
    song_df = utils.load_songs(cfg['input'])
    song_df = nlp.get_song_stats(song_df)
    album_df = songs2albums(song_df)
    band_df = albums2bands(album_df)
    genre_cols = [c for c in band_df.columns if 'genre_' in c]
    columns = genre_cols + ['word_count', 'word_count_uniq', 'word_rate', 'word_rate_uniq']
    genre_df = song_df[columns].copy()
    plots = cfg.get('plots', {})

    for key, value in plots.items():
        data_arg = key.split('_')[0] + '_df'
        line = f"plot_{key}({data_arg}, os.path.join(output, value))"
        print(line)
        exec(line)


if __name__ == '__main__':
    main()
