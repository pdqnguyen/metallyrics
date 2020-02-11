#! /usr/bin/env python

import os
import re
import time
import pickle as pkl
from argparse import ArgumentParser
from traceback import print_exc

from utils import scrape_html


BASEURL = 'http://www.darklyrics.com'
CRAWLDELAY = 10


def get_album_lyrics(album_name, album_url):
    soup = scrape_html(album_url)
    if soup is None:
        return []
    songs = soup.find_all('h3')
    album_lyrics = {}
    for song in songs:
        song_name = song.text
        song_lyrics = []
        for line in song.next_siblings:
            if line.name is None:
                s = str(line).strip().replace('"', '')
                if s != '':
                    song_lyrics.append(s)
            elif line.name != 'br' and line.name != 'i':
                break
        album_lyrics[song_name] = song_lyrics
    return album_lyrics

def get_band_lyrics(band_name, verbose=False):
    band_name = band_name.replace(' ', '').lower()
    band_url = BASEURL + '/' + band_name[0] + '/' + band_name + '.html'
    soup = scrape_html(band_url)
    if soup is None:
        return []
    albums_html = soup.find_all('div', attrs={'class': 'album'})
    band_lyrics = {}
    for i, album in enumerate(albums_html):
        try:
            album_name = album.find('strong').text.replace('"', '')
        except Exception:
            continue
        album_url = BASEURL + album.find('a', href=True)['href'][2:-2]
        if verbose:
            print(album_name, album_url)
        album_lyrics = get_album_lyrics(album_name, album_url)
        band_lyrics[album_name] = album_lyrics
        if i < len(albums_html) - 1:
            time.sleep(CRAWLDELAY)
    return band_lyrics


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("bands", nargs="*")
    parser.add_argument("-f", "--file")
    parser.add_argument("-o", "--outdir")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    bands = args.bands
    file = args.file
    outdir = args.outdir
    verbose = args.verbose
    if file is not None:
        with open(file) as f:
            bands = [line.rstrip() for line in f.readlines()]
    for band in bands:
        if verbose:
            print(band)
        try:
            basename = band.lower().replace(' ', '')
            lyrics = get_band_lyrics(basename, verbose=verbose)
            if outdir is not None and len(lyrics) > 0:
                filename = os.path.join(outdir, basename + '.pkl')
                with open(filename, 'wb') as f:
                    pkl.dump(lyrics, f, protocol=pkl.HIGHEST_PROTOCOL)
            else:
                print(lyrics)
        except KeyboardInterrupt:
            raise
        except Exception:
            print_exc()
