from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

BASEURL = 'http://www.darklyrics.com'

def get_album_lyrics(album_name, album_url):
    with urlopen(album_url) as album_page:
        soup = BeautifulSoup(album_page, 'html.parser')
    songs = soup.find_all('h3')
    album_lyrics = {}
    for song in songs:
        song_name = song.text
        song_lyrics = []
        for line in song.next_siblings:
            if line.name is None:
                s = str(line).replace('\n', '').replace('"', '')
                if s != '':
                    song_lyrics.append(s)
            elif line.name != 'br' and line.name != 'i':
                break
        album_lyrics[song_name] = song_lyrics
    return album_lyrics

def get_band_lyrics(band_name, verbose=False):
    band_url = BASEURL + '/' + band_name[0] + '/' + band_name + '.html'
    with urlopen(band_url) as band_page:
        soup = BeautifulSoup(band_page, 'html.parser')
    albums_html = soup.find_all('div', attrs={'class': 'album'})
    band_lyrics = {}
    for album in albums_html:
        album_name = album.find('strong').text.replace('"', '')
        album_url = BASEURL + album.find('a', href=True)['href'][2:-2]
        if verbose:
            print(album_name, album_url)
        album_lyrics = get_album_lyrics(album_name, album_url)
        band_lyrics[album_name] = album_lyrics
    return band_lyrics

if __name__ == '__main__':
    import sys
    bands_list = sys.arv[1]
    with open(bands_list) as f:
        band_names = f.readlines()
        for band_name in band_names:
            band_lyrics = get_band_lyrics(band_name)