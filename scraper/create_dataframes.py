import glob
import os
import json
from argparse import ArgumentParser
import pandas as pd
from tqdm import trange


def create_reviews_df(bands, filename):
    rows = []
    t = trange(len(bands), desc="looping over artists")
    for band, i in zip(bands, t):
        band_data = dict(
            band_name=band['name'],
            band_id=band['id'],
            band_url=band['url']
        )
        band_info = {'band_' + key.lower().replace(' ', '_'): value
                     for key, value in band['info'].items()}
        band_data.update(band_info)
        for album in band['albums']:
            album_data = dict(
                album_name=album['name'],
                album_type=album['type'],
                album_year=album['year'],
                album_review_num=album['review_num'],
                album_review_avg=album['review_avg'],
                album_url=album['url'],
                album_reviews_url=album['reviews_url'],
                album_song_names=album['song_names']
            )
            for review in album['reviews']:
                review_data = {'review_' + key: value
                               for key, value in review.items()}
                row_data = band_data.copy()
                row_data.update(album_data)
                row_data.update(review_data)
                row = pd.Series(row_data)
                rows.append(row)
    df_reviews = pd.DataFrame(rows)
    df_reviews.to_csv(filename, index=False)
    return


def create_songs_df(bands, filename):
    song_keys = ['name', 'length', 'url', 'darklyrics', 'darklyrics_url']
    rows = []
    t = trange(len(bands), desc="looping over artists")
    for band, i in zip(bands, t):
        band_data = dict(
            band_name=band['name'],
            band_id=band['id'],
            band_url=band['url']
        )
        band_info = {'band_' + key.lower().replace(' ', '_'): value
                     for key, value in band['info'].items()}
        band_data.update(band_info)
        for album in band['albums']:
            album_data = dict(
                album_name=album['name'],
                album_type=album['type'],
                album_year=album['year'],
                album_review_num=album['review_num'],
                album_review_avg=album['review_avg'],
                album_url=album['url'],
                album_reviews_url=album['reviews_url'],
            )
            for song in album['songs']:
                song_data = {'song_' + key: song.get(key) for key in song_keys}
                row_data = band_data.copy()
                row_data.update(album_data)
                row_data.update(song_data)
                row = pd.Series(row_data)
                rows.append(row)
    df_songs = pd.DataFrame(rows)
    df_songs.to_csv(filename, index=False)
    return


def main():
    parser = ArgumentParser()
    parser.add_argument("which", help="'reviews' or 'songs'")
    parser.add_argument("input", nargs="+")
    parser.add_argument("output")
    args = parser.parse_args()
    input = []
    for inp in args.input:
        if '*' in inp:
            input.extend(glob.glob(inp))
        else:
            input.append(inp)
    bands = [json.load(open(inp, 'r')) for inp in input]
    if args.which.strip().lower() == 'reviews':
        create_reviews_df(bands, args.output)
    if args.which.strip().lower() == 'songs':
        create_songs_df(bands, args.output)
    return


if __name__ == '__main__':
    main()