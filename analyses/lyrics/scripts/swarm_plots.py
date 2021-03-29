"""
Produce swarm plot illustrating the vocabularies of bands.
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import get_path_collection_extents
import seaborn as sns

import lyrics_utils as utils


plt.style.use('seaborn')
plt.rcParams.update(
    {
        'figure.facecolor': 'white',
        'figure.titlesize': 25,
    }
)


def songs2bands(data):
    out = pd.concat(
        (
            data.groupby('band_id')['band_name'].first(),
            data.groupby(['band_id', 'album_name'])['album_review_num'].first().groupby('band_id').sum(),
            data.groupby('band_id')['song_words'].sum(),
            data.groupby('band_id')[['word_count']].sum()
        ),
        axis=1
    )
    out.columns = ['name', 'reviews', 'words', 'word_count']
    return out


def get_bboxes(sc, ax):
    """
    Copied from:
    https://stackoverflow.com/questions/55005272/get-bounding-boxes-of-individual-elements-of-a-pathcollection-from-plt-scatter
    Returns a list of bounding boxes in data coordinates for a scatter plot
    """
    ax.figure.canvas.draw() # need to draw before the transforms are set.
    transform = sc.get_transform()
    transOffset = sc.get_offset_transform()
    offsets = sc._offsets
    paths = sc.get_paths()
    transforms = sc.get_transforms()

    if not transform.is_affine:
        paths = [transform.transform_path_non_affine(p) for p in paths]
        transform = transform.get_affine()
    if not transOffset.is_affine:
        offsets = transOffset.transform_non_affine(offsets)
        transOffset = transOffset.get_affine()

    if isinstance(offsets, np.ma.MaskedArray):
        offsets = offsets.filled(np.nan)

    bboxes = []

    if len(paths) and len(offsets):
        if len(paths) < len(offsets):
            # for usual scatters you have one path, but several offsets
            paths = [paths[0]]*len(offsets)
        if len(transforms) < len(offsets):
            # often you may have a single scatter size, but several offsets
            transforms = [transforms[0]]*len(offsets)

        for p, o, t in zip(paths, offsets, transforms):
            result = get_path_collection_extents(
                transform.frozen(), [p], [t],
                [o], transOffset.frozen())
            bboxes.append(result.transformed(ax.transData.inverted()))

    return bboxes


def plot_swarm(data, names):
    """
    Swarm plot with text labels forced to fit in data makers
    """
    fig = plt.figure(figsize=(25, 10))
    ax = sns.swarmplot(x=data, size=30, zorder=1)

    # Get bounding boxes of scatter points
    cs = ax.collections[0]
    boxes = get_bboxes(cs, ax)

    # Add text to circles
    for i, box in enumerate(boxes):
        x = box.x0 + box.width / 2
        y = box.y0 + box.height / 2
        s = names.iloc[i].replace(' ', '\n')
        txt = ax.text(x, y, s, va='center', ha='center')

        # Shrink font size until text fits completely in circle
        for fs in range(10, 1, -1):
            txt.set_fontsize(fs)
            tbox = txt.get_window_extent().transformed(ax.transData.inverted())
            if (
                    abs(tbox.width) < np.cos(0.5) * abs(box.width)
                    and abs(tbox.height) < np.cos(0.5) * abs(box.height)
            ):
                break

    ax.xaxis.tick_top()
    ax.set_xlabel('')

    return fig


def uniq_first_words(x, num_words):
    return len(set(x[:num_words]))


def plot_band_words(data, filepath, num_bands=None, num_words=None):
    data_short = data[data['word_count'] > num_words].copy()
    top_reviewed_bands = data_short.sort_values('reviews')['name'][-num_bands:]
    data_short = data_short.loc[top_reviewed_bands.index]
    data_short['unique_first_words'] = data_short['words'].apply(uniq_first_words, args=(num_words,))
    data_short = data_short.sort_values('unique_first_words')
    fig = plot_swarm(data_short['unique_first_words'], data_short['name'])
    fig.suptitle(f"# of unique words in first {num_words:,.0f} of artist's lyrics", fontsize=25)
    plt.savefig(filepath)


def plot_band_word_rate(data, filepath, num_bands=None, num_words=None):
    data_short = data[data['word_count'] > num_words].copy()
    top_reviewed_bands = data_short.sort_values('reviews')['name'][-num_bands:]
    data_short = data_short.loc[top_reviewed_bands.index]
    data_short['unique_first_words'] = data_short['words'].apply(uniq_first_words, args=(num_words,))
    data_short = data_short.sort_values('unique_first_words')
    fig = plot_swarm(data_short['unique_first_words'], data_short['name'])
    fig.suptitle(f"# of unique words in first {num_words:,.0f} of artist's lyrics", fontsize=25)
    plt.savefig(filepath)


def main():
    cfg = utils.get_config()
    output = cfg['output']
    if not os.path.exists(output):
        os.mkdir(output)
    song_df = utils.load_songs(cfg['input'])
    song_df['word_count'] = song_df['song_words'].apply(len)
    band_df = songs2bands(song_df)
    print(len(band_df))

    plots = cfg.get('plots', {})
    for key, value in plots.items():
        plot_func = eval(f"plot_{key}")
        filepath = os.path.join(output, value.pop('filename'))
        plot_func(band_df, filepath, **value)


if __name__ == '__main__':
    main()