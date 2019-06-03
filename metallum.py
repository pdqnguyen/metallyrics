from urllib.request import urlopen
from urllib.parse import quote_plus, unquote
import pandas as pd
import time
import re
import os
import pickle

from utils import scrape_html, get_genres

BASEURL = 'https://www.metal-archives.com/'
CRAWLDELAY = 3

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
        strkey = key.text.replace(':', '')
        if strkey in key_dict.keys():
            info_key = key_dict[strkey]
            info_val = val.text
            if info_key == 'formation':
                info[info_key] = int(info_val)
            elif info_key == 'genres':
                info[info_key] = get_genres(info_val)
            elif info_key == 'years':
                years_str = re.findall('[0-9\-]+', info_val)
                years = []
                for s in years_str:
                    split = s.split('-')
                    years.append(tuple(map(lambda x: None if x == '' else int(x), split)))
                info[info_key] = years
            elif info_key == 'themes':
                info[info_key] = info_val.lower().split(', ')
            else:
                info[info_key] = info_val
    return info

def get_song_dict(album_url):
    """
    Get song names and corresponding song id numbers for a single album on metallum.
    
    Paramters
    ---------
    album_url : str
    
    Returns
    -------
    song_dict : dict
    """
    
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

class Song(object):
    """
    Single song with lyrics.
    
    Attributes
    ----------
    name : str
    lyrics : list of strings
    """
    
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
    """
    Class object for albums, containing songs, reviews, and lyrics.
    
    Attributes
    ----------
    name : str
    songs : list of metallum.Song objects
    song_names : list of strings
    reviews : list
    lyrics : list of strings
    rating : float
    """
    
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
        rtg = (tot - 50*n) / 50
        # rtg = np.sign(rtg) * np.log1p(abs(rtg))
        return rtg
    
    @classmethod
    def fetch(cls, band_name, album_name, album_id, verbose=False):
        if verbose:
            print('fetching album ' + album_name)
        band_name_url = quote_plus(band_name.replace(' ','_'))
        album_name_url = quote_plus(album_name.replace(' ','_'))
        # Get song names and ids
        album_url = BASEURL + 'albums/' + band_name_url + '/' + album_name_url + '/' + album_id
        song_dict = get_song_dict(album_url)
        # Fetch song lyrics
        songs = []
        for song_name, song_id in song_dict.items():
            break
            if verbose:
                print('fetching song ' + song_name)
            try:
                song = Song.fetch(song_name, song_id)
            except AssertionError:
                continue
            else:
                songs.append(song)
        # Get reviews
        reviews_url = album_url.replace('albums', 'reviews')
        reviews = get_reviews(reviews_url)
        return cls(album_name, songs=songs, reviews=reviews)

class Band(object):
    """
    Object class for storing albums and band info for a single band.
    
    Attributes
    ----------
    name : str
    albums : list of metallum.Album objects
    album_names : list of strings
    formation : int
    genres : list of strings
    label : str
    location : str
    origin : str
    themes : list of strings
    years : list of int tuples
    """
    
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
    def fetch(cls, band_name, band_id, verbose=False):
        disco_url = BASEURL + 'band/discography/id/' + band_id + '/tab/all'
        disco_html = scrape_html(disco_url)
        assert disco_html is not None
        rows = disco_html.find('table').find_all('tr')
        albums = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) == 4:
                url = cols[0].find('a', attrs={'class': 'album'})
                rating = re.search('([0-9]+)(?=\%)', cols[3].text)
                if url is not None and rating is not None:
                    url_end = re.search('(?<=albums\/).*', url['href']).group(0)
                    _, album_name, album_id = url_end.split('/')
                    album = Album.fetch(band_name, album_name, album_id, verbose=verbose)
                    albums.append(album)
        new = cls(band_name, albums=albums)
        band_name_url = quote_plus(band_name.replace(' ','_'))
        band_url = BASEURL + 'bands/' + band_name_url + '/' + band_id
        band_info = get_band_info(band_url)
        for key, val in band_info.items():
            setattr(new, key, val)
        return new
    
    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            new = pickle.load(f)
        return new
    
    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_band(band_name, band_id, out_dir, verbose=False):
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
        band = Band.fetch(band_name, band_id, verbose=verbose)
    except:
        print('{} failed.'.format(band_name))
        return
    band_name_url = quote_plus(band_name.replace(' ', '_'))
    file = os.path.join(out_dir, band_name_url + '_' + band_id + '.pkl')
    band.save(file)
    t1 = time.time()
    if verbose:
        print('{} complete: {:.0f} s'.format(band_name, t1 - t0))

def get_bands(csv_name, out_dir, verbose=False):
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
    band_df = pd.read_csv(csv_name, names=['name', 'id'])
    for idx, row in band_df.iterrows():
        band_name = row['name']
        band_id = str(row['id'])
        print(band_name)
        get_band(band_name, band_id, out_dir, verbose=verbose)
    dt = time.time() - t0
    print('Complete: {} minutes'.format(dt / 60.))

if __name__ == '__main__':
    from metallum import Band
    import sys
    get_bands(sys.argv[1], sys.argv[2])