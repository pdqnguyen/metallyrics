#! /usr/bin/env python

import os
import re
import json
import time
import traceback
from argparse import ArgumentParser
from urllib.request import urlopen
from urllib.parse import quote_plus, unquote_plus
from urllib.request import urlopen, HTTPError, Request
from bs4 import BeautifulSoup
import pandas as pd


USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'
BASEURL = 'https://www.metal-archives.com/'
CRAWLDELAY = 3


def scrape_html(url, user_agent=USER_AGENT):
    headers = {'User-Agent': user_agent}
    req = Request(url=url, headers=headers)
    try:
        page = urlopen(req).read()
        soup = BeautifulSoup(page, 'html.parser')
    except HTTPError:
        print('url not found: ' + url)
        soup = None
    return soup


def get_band_info(url):
    """
    Get info for a band on metallum.
    
    Parameters
    ----------
    url : str
    
    Returns
    -------
    info : dict
    """
    
    soup = scrape_html(url)
    raw_info = soup.find('div', attrs={'id': 'band_stats'})
    keys = raw_info.find_all('dt')
    vals = raw_info.find_all('dd')
    info = {key.text.replace(':', ''): val.text for key, val in zip(*(keys, vals))}
    info = {}
    key_dict = {
        'Country of origin': 'origin',
        'Current label': 'label',
        'Formed in': 'formation',
        'Genre': 'genres',
        'Location': 'location',
        'Lyrical themes': 'themes',
        'Years active': 'years'
    }
    for key, val in zip(*(keys, vals)):
        info[key.text.replace(':', '')] = val.text
#         strkey = key.text.replace(':', '')
#         if strkey in key_dict.keys():
#             info_key = key_dict[strkey]
#             info_val = val.text
#             if info_key == 'formation':
#                 info[info_key] = int(info_val)
#             elif info_key == 'genres':
#                 info[info_key] = get_genres(info_val)
#             elif info_key == 'years':
#                 years_str = re.findall('[0-9\-]+', info_val)
#                 years = []
#                 for s in years_str:
#                     split = s.split('-')
#                     years.append(tuple(map(lambda x: None if x == '' else int(x), split)))
#                 info[info_key] = years
#             elif info_key == 'themes':
#                 info[info_key] = info_val.lower().split(', ')
#             else:
#                 info[info_key] = info_val
    return info


def get_song_urls(album_url):
    """
    Get song names and corresponding song id numbers for a single album on metallum.
    
    Paramters
    ---------
    album_url : str
    
    Returns
    -------
    out : dict
    """
    
    soup = scrape_html(album_url)
    table = soup.find('table', attrs={'class': 'display table_lyrics'})
    urls = []
    for tr in table.find_all('tr'):
        name_cell = tr.find('td', attrs={'class': 'wrapWords'})
        link = tr.find('a')
        if name_cell is not None and link is not None:
            name = name_cell.text.strip()
            id_ = link['name']
            url = os.path.join(BASEURL, 'release/ajax-view-lyrics/id', id_)
            urls.append((name, url))
    return urls


def get_reviews(reviews_url):
    """
    Get reviews (out of 100) for an album.
    
    Parameters
    ----------
    reviews_url : str
    
    Returns
    -------
    reviews : list
    """    
    
    soup = scrape_html(reviews_url)
    titles = soup.find_all(attrs={'class': 'reviewTitle'})
    reviews = []
    for title in titles:
        match = re.search('([0-9]+)(?=\%)', title.text)
        if match:
            try:
                pct = int(match.group(0))
            except ValueError:
                pass
            else:
                reviews.append(pct)
    return reviews


def get_genres(genre_string):
    pattern = '[a-zA-Z\-]+'
    genres = [match.lower() for match in re.findall(pattern, genre_string)]
    genres = sorted(set(genres))
    if 'metal' in genres:
        genres.remove('metal')
    return genres


class Song(object):
    """
    Single song with lyrics.
    """
    
    def __init__(self, name, lyrics=[]):
        self.name = name
        self.lyrics = lyrics
    
    @classmethod
    def fetch(cls, name, url, verbose=False):
        if verbose:
            print("fetching song " + name)
        soup = scrape_html(url)
        time.sleep(CRAWLDELAY)
        lyrics = soup.text.strip()
        if lyrics.lower() == '(lyrics not available)':
            raise ValueError(name + " lyrics not available")
        elif lyrics.lower() == '(instrumental)':
            lyrics = ''
        return cls(name, lyrics)

    def to_json(self):
        return dict(name=self.name, lyrics=self.lyrics)


class Album(object):
    """
    Class object for albums, containing songs, reviews, and lyrics.
    """
    
    def __init__(self, name, songs=[], reviews=[]):
        self.name = name
        self.songs = songs
        self.reviews = reviews
        self.lyrics = [song.lyrics for song in self.songs]
    
    @property
    def song_names(self):
        return [song.name for song in self.songs]
    
    @property
    def rating(self):
        tot = sum(self.reviews)
        n = len(self.reviews)
        rtg = (tot - 50*n) / 50
        # rtg = np.sign(rtg) * np.log1p(abs(rtg))
        return rtg
    
    @classmethod
    def fetch(cls, name, url, verbose=False):
        if verbose:
            print("fetching album " + name)
        # Get song names and ids
        song_urls = get_song_urls(url)
        time.sleep(CRAWLDELAY)
        # Fetch song lyrics
        songs = []
        for song_name, song_url in song_urls:
            try:
                song = Song.fetch(song_name, song_url, verbose=verbose)
            except KeyboardInterrupt:
                raise
            except:
                if verbose:
                    traceback.print_exc()
                print(song_name + " failed")
            else:
                songs.append(song)
        # Get reviews
        reviews_url = url.replace('albums', 'reviews')
        reviews = get_reviews(reviews_url)
        time.sleep(CRAWLDELAY)
        return cls(name, songs=songs, reviews=reviews)

    def to_json(self):
        return dict(
            name=self.name,
            songs=[song.to_json() for song in self.songs],
            song_names=self.song_names,
            rating=self.rating
        )


class Band(object):
    """
    Object class for storing albums and band info for a single band.
    """
    
    def __init__(self, name, albums=[], info={}):
        self.name = name
        self.albums = albums
        self.info = info
    
    @property
    def album_names(self):
        return [album.name for album in self.albums]
    
    @classmethod
    def fetch(cls, name, url, verbose=False):
        if verbose:
            print("fetching band " + name)
        # Get band information
        info = get_band_info(url)
        # Scrape discography page for album URLs
        id_ = os.path.split(url)[-1]
        disco_url = os.path.join(BASEURL, 'band/discography/id', id_, 'tab/all')
        disco_html = scrape_html(disco_url)
        if disco_html is None:
            raise ValueError("no discography found")
        time.sleep(CRAWLDELAY)
        rows = disco_html.find('table').find_all('tr')
        albums = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 4:
                album_link = cols[0].find('a', attrs={'class': 'album'})
                rating = re.search('([0-9]+)(?=\%)', cols[3].text)
                if album_link is not None and rating is not None:
                    album_url = album_link['href']
                    album_name = unquote_plus(album_url.rstrip('/').split('/')[-2])
                    try:
                        album = Album.fetch(album_name, album_url, verbose=verbose)
                    except KeyboardInterrupt:
                        raise
                    except:
                        if verbose:
                            traceback.print_exc()
                        print(album_name + ' failed')
                    else:
                        albums.append(album)
        new = cls(name, albums=albums, info=info)
        time.sleep(CRAWLDELAY)
        return new

    def save(self, file):
        data = dict(
            name=self.name,
            albums=[album.to_json() for album in self.albums],
            album_names=self.album_names,
            info=self.info
        )
        with open(file, 'w') as f:
            json.dump(data, f)
        return


def get_band(name, url, out_dir, verbose=False):
    """
    Scrape metallum for a band and save as an instance of Band.
    
    Parameters
    ----------
    band_name : str
    band_id : str
    out_dir : str
    verbose : {False, True}, optional
    """
    
    t0 = time.time()
    try:
        band = Band.fetch(name, url, verbose=verbose)
    except KeyboardInterrupt:
        raise
    except:
        if verbose:
            traceback.print_exc()
        print('{} failed.'.format(name))
        return
    filename = '_'.join(url.rstrip('/').split('/')[-2:]) + '.json'
    file = os.path.join(out_dir, filename)
    band.save(file)
    t1 = time.time()
    if verbose:
        print('{} complete: {:.0f} s'.format(name, t1 - t0))
    return


def main(csv_name, out_dir, verbose=False):
    """
    Download and save Band object for every band/id pair in csv.
    
    Parameters
    ----------
    csv_name : str
    out_dir : str
    """
    
    t0 = time.time()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    band_df = pd.read_csv(csv_name)
    for idx, row in band_df.iterrows():
        name = row['name']
        id_ = str(row['id'])
        url = os.path.join(BASEURL, 'bands', quote_plus(name), id_)
        get_band(name, url, out_dir, verbose=verbose)
    dt = time.time() - t0
    print('Complete: {} minutes'.format(dt / 60.))
    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("ids")
    parser.add_argument("outdir")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()
    main(args.ids, args.outdir, verbose=args.verbose)
