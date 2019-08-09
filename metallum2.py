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
        return self.albums[item]
    
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
                rtg = re.findall('[0-9]+', cols[3].text)
                if url is not None and len(rtg) > 0:
                    url_end = re.search('(?<=albums\/).*', url['href']).group(0)
                    a_name = url_end.split('/')[1]
                    try:
                        a_year = int(cols[2].text)
                    except:
                        a_year = 0
                    a_num = int(rtg[0])
                    a_avg = int(rtg[1])
                    a = (a_name, a_year, a_num, a_avg)
                    albums.append(a)
        if len(albums) > 0:
            albums = pd.DataFrame(albums)
            albums.columns = ['album', 'year', 'numrev', 'avgrev']
        else:
            albums = None
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
        print('{} {} failed.'.format(band_name, band_id))
    else:
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
    band_df = pd.read_csv(csv_name)
    for idx, row in band_df.iterrows():
        band_name = row['name']
        band_id = str(row['id'])
        print(band_name)
        get_band(band_name, band_id, out_dir, verbose=verbose)
        time.sleep(CRAWLDELAY)
    dt = time.time() - t0
    print('Complete: {} minutes'.format(dt / 60.))

if __name__ == '__main__':
    from metallum2 import Band
    import sys
    get_bands(sys.argv[1], sys.argv[2])