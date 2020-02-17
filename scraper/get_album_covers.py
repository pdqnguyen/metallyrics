from urllib.request import urlopen, HTTPError
from urllib.parse import quote_plus, unquote
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import re
import os
import pickle
import glob

BASEURL = 'https://www.metal-archives.com/albums/'
CRAWLDELAY = 5

BANDS_DIR = 'bands'
IMG_DIR = 'imgs'


def get_album_cover(url):
    with urlopen(url) as page:
        soup = BeautifulSoup(page, 'html.parser')
    img_div = soup.find('div', attrs={'class': 'album_img'})
    img_url = img_div.find('a', href=True)['href']
    img = requests.get(img_url).content
    return img


band_files = glob.glob('{}/*.pkl'.format(BANDS_DIR))
for f in band_files[]:
    try:
        band = pickle.load(open(f, 'rb'))
        band_name = quote_plus(band.name.replace(' ', '_'))
    except:
        continue
    if band.albums is not None:
        album_names = band.albums.album
        for album_name in album_names:
            album_name = quote_plus(album_name)
            url = BASEURL + band_name + '/' + album_name
            print(band_name, album_name, url)
            try:
                img = get_album_cover(url)
            except:
                continue
            filename = '{}/{}-{}.jpg'.format(IMG_DIR, band_name, album_name)
            try:
                with open(filename, 'wb') as f:
                    f.write(img)
            except:
                pass
            time.sleep(CRAWLDELAY)
