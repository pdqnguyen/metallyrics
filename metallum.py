from urllib.request import urlopen
from urllib.parse import quote_plus
import pandas as pd
import time
import re

from utils import scrape_html

BASEURL = 'https://www.metal-archives.com/'
CRAWLDELAY = 3

def get_band_info(url):
    soup = scrape_html(url)
    name = soup.find('h1', attrs={'class': 'band_name'}).text
    raw_info = soup.find('div', attrs={'id': 'band_stats'})
    keys = raw_info.find_all('dt')
    vals = raw_info.find_all('dd')
    info = {key.text.replace(':', ''): val.text for key, val in zip(*(keys, vals))}
    return info

def get_song_dict(album_url):
    soup = scrape_html(album_url)
    song_table = soup.find('table', attrs={'class': 'display table_lyrics'})
    song_table_df = pd.read_html(str(song_table))[0]
    song_table_df = song_table_df[:-1:2].reset_index()
    song_names = list(song_table_df[1].values)
    song_ids = []
    for tr in song_table.find_all('tr'):
        if tr.find('a') is not None:
            song_ids.append(tr.find('a')['name'])
    song_dict = dict(zip(*(song_names, song_ids)))
    return song_dict

def get_reviews(reviews_url):
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

class Song(object):
    def __init__(self, name, lyrics=[]):
        self.name = name
        self.lyrics = lyrics
    
    @classmethod
    def fetch(cls, name, song_id):
        song_url = BASEURL + 'release/ajax-view-lyrics/id/' + song_id
        with urlopen(song_url) as conn:
            soup = scrape_html(song_url)
        time.sleep(CRAWLDELAY)
        lyrics = soup.text.lower().split()
        assert ' '.join(lyrics) != '(lyrics not available)'
        assert ' '.join(lyrics) != '(instrumental)'
        return cls(name, lyrics)

class Album(object):
    def __init__(self, name, songs=[], reviews=[]):
        self.name = name
        self.songs = songs
        self.reviews = reviews
        self.lyrics = [word for song in self.songs for word in song.lyrics]
    
    def __getitem__(self, item):
        index = self.song_names.index(item)
        return self.songs[index]
    
    @property
    def song_names(self):
        return [song.name for song in self.songs]
    
    @property
    def rating(self):
        tot = sum(self.reviews)
        n = len(self.reviews)
        rtg = (total - 50*n) / 50
        # rtg = np.sign(rtg) * np.log1p(abs(rtg))
        return rtg
    
    @classmethod
    def fetch(cls, band_name, album_name):
        print('fetching album ' + album_name)
        band_name_url = quote_plus(band_name.replace(' ','_'))
        album_name_url = quote_plus(album_name.replace(' ','_'))
        # Get song names and ids
        album_url = BASEURL + 'albums/' + band_name_url + '/' + album_name_url
        song_dict = get_song_dict(album_url)
        # Fetch song lyrics
        songs = []
        for name_, id_ in song_dict.items():
            print('fetching song ' + name_)
            try:
                song = Song.fetch(name_, id_)
            except AssertionError:
                continue
            else:
                songs.append(song)
        # Get reviews
        reviews_url = album_url.replace('albums', 'reviews')
        reviews = get_reviews(reviews_url)
        return cls(album_name, songs=songs, reviews=reviews)

class Band(object):
    def __init__(self, name, albums=[]):
        self.name = name
        self.albums = albums
    
    def __getitem__(self, item):
        try:
            album_name, song_name = item
        except:
            album_name = item
            index = self.album_names.index(album_name)
            album = self.albums[index]
            return album
        else:
            index = self.album_names.index(album_name)
            album = self.albums[index]
            song = album.__getitem__(song_name)
            return song
    
    @property
    def album_names(self):
        return [album.name for album in self.albums]
    
    @classmethod
    def fetch(cls, band_name, band_id):
        disco_url = BASEURL + 'band/discography/id/' + band_id + '/tab/all'
        soup = scrape_html(disco_url)
        disco = pd.read_html(str(soup.find('table')))[0]
        disco = disco[~pd.isnull(disco['Reviews'])]
        disco = disco[disco['Type'] == 'Full-length']
        albums = [Album.fetch(band_name, album_name) for album_name in disco['Name']]
        return cls(band_name, albums=albums)