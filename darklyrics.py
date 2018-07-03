from utils import scrape_html
import re
import time

BASEURL = 'http://www.darklyrics.com'
CRAWLDELAY = 10

def get_band_names(letter):
    band_names_url = BASEURL + '/' + letter.lower() + '.html'
    soup = scrape_html(band_names_url)
    band_names = []
    urls = [line['href'] for line in soup.find_all('a', href=True)]
    for url in urls:
        pattern = '(?<=' + letter.lower() + '\/)([a-z0-9]+)(?=\.html)'
        match = re.search(pattern, url)
        if match is not None:
            band_names.append(match.group(0))
    return band_names

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
                s = str(line).replace('\n', '').replace('"', '')
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
    for album in albums_html:
        album_name = album.find('strong').text.replace('"', '')
        album_url = BASEURL + album.find('a', href=True)['href'][2:-2]
        if verbose:
            print(album_name, album_url)
        album_lyrics = get_album_lyrics(album_name, album_url)
        band_lyrics[album_name] = album_lyrics
        time.sleep(CRAWLDELAY)
    return band_lyrics

if __name__ == '__main__':
    import sys
    import time
    bands_list = sys.arv[1]
    with open(bands_list) as f:
        band_names = f.readlines()
        for band_name in band_names:
            band_lyrics = get_band_lyrics(band_name)
            time.sleep(CRAWLDELAY)