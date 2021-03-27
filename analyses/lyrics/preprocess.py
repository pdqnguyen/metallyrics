import re
import pandas as pd
import yaml
from argparse import ArgumentParser

from nlp import tokenize


MIN_ENGLISH = 0.6  # Songs with lower English word fraction are filtered out. Default is a little higher than 50% to
# include songs with translations, whose lyrics typically include original and translated text


def filter_english(data, min_english=MIN_ENGLISH):
    """
    Remove songs that are mostly non-English
    """
    rows = []
    song_words = []
    for i, row in data.iterrows():
        text = row.song_darklyrics.strip()
        words = tokenize(text)
        english_words = tokenize(text, english_only=True)
        is_english = len(english_words) > min_english * len(words)
        if is_english:
            rows.append(row)
            song_words.append(english_words)
    print('Non-English songs removed: ', len(data) - len(rows))
    data = pd.DataFrame(rows, columns=data.columns)
    data['song_words'] = song_words
    return data


def filter_copyright(data):
    """
    Remove songs that were copyright claimed
    """
    copyrighted = data.song_darklyrics.str.contains('lyrics were removed due to copyright holder\'s request')
    print('Songs with lyrics removed: ', len(data[copyrighted]))
    data = data[~copyrighted]
    return data


def create_genre_columns(data):
    song_genres = data.band_genre.apply(process_genre)
    genres = sorted(set(song_genres.sum()))
    cols = [f'genre_{genre}' for genre in genres]
    for genre, col in zip(genres, cols):
        data[col] = song_genres.apply(lambda x: int(genre in x))
    return data, cols


def process_genre(genre):
    """
    Find words (including hyphenated words) not in parentheses
    """
    out = re.findall('[\w\-]+(?![^(]*\))', genre.lower())
    out = [s for s in out if s != 'metal']
    return out


def get_top_genres(data, min_pct):
    """
    Get genre tags appearing in more than min_pct rows of data
    """
    isolated = (data.sum(axis=1) == 1)
    isolated_cols = sorted(set(data[isolated].idxmax(axis=1)))
    top_cols = [col for col in isolated_cols if data[col][isolated].mean() >= min_pct]
    top_genres = [re.sub(r"^genre\_", "", col) for col in top_cols]
    return top_genres


def reduce_dataset(data, cols, min_pct):
    """
    Reduce genre label space to genres appearing in at least min_pct of rows in data
    """
    top_genres = get_top_genres(data[cols], min_pct)
    out = data.copy()
    drop_cols = [col for col in data.columns if ('genre_' in col) and (re.sub(r"^genre\_", "", col) not in top_genres)]
    out.drop(drop_cols, axis=1, inplace=True)
    return out, top_genres


def ml_dataset(data, genres):
    """
    Generate `pd.DataFrame` with only lyrics and genre columns
    """
    out = pd.DataFrame(index=range(data.shape[0]), columns=['lyrics'] + genres)
    out['lyrics'] = data['song_darklyrics'].reset_index(drop=True)
    out[genres] = data[[f"genre_{genre}" for genre in genres]].reset_index(drop=True)
    return out


def main(config):
    with open(config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    df = pd.read_csv(cfg['input'], low_memory=False)
    df = df[~df.song_darklyrics.isnull()]
    df = df[df.song_darklyrics.str.strip().apply(len) > 0]
    print('Songs: ', len(df))
    # Filter data
    df = filter_english(df)
    df = filter_copyright(df)
    print('Creating genre columns')
    df, genre_cols = create_genre_columns(df)
    print('Creating reduced datasets')
    for d in cfg['datasets']:
        df_r, top_genres = reduce_dataset(df, genre_cols, d['min_pct'])
        df_r_ml = ml_dataset(df, top_genres)
        df_r.to_csv(d['output'], index=False)
        df_r_ml.to_csv(d['ml-output'], index=False)
    print('Done')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    main(args.config)