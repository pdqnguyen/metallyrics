"""
Populate the output of metallum.py with lyrics fetched from Dark Lyrics.
"""


import os
import glob
import json
import logging
import re
import sys
import time
from argparse import ArgumentParser
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urljoin

from unidecode import unidecode
from bs4 import BeautifulSoup
from tqdm import trange


USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'
CRAWL_DELAY = 10  # Time between accessing pages; be nice, don't lower this number
BASEURL = 'http://www.darklyrics.com'
LETTERS = 'abcdefghijklmnopqrstuvwxyz'
LOGFILE = os.path.abspath('scraper_darklyrics.log')


class ScrapingError(Exception):
    """Handle errors in scraping web pages
    """
    pass


def configure_logger(
        logfile=LOGFILE, file_level=logging.INFO, stream_level=logging.ERROR):
    if os.path.exists(logfile):
        os.remove(logfile)
    logger = logging.getLogger()
    logger.setLevel(file_level)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s --- %(message)s'))
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s --- %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def scrape_html(url, user_agent=USER_AGENT, crawl_delay=CRAWL_DELAY):
    headers = {'User-Agent': user_agent}
    try:
        req = Request(url=url, headers=headers)
        page = urlopen(req).read()
        soup = BeautifulSoup(page, 'html.parser')
    except (HTTPError, ValueError):
        raise ScrapingError("invalid url " + str(url))
    time.sleep(crawl_delay)
    return soup


def get_album_lyrics(url):
    """Fetch all lyrics for an album
    """
    logger.info("fetching lyrics from " + url)
    soup = scrape_html(url)   # Album page contains all song lyrics
    if soup is None:
        return []
    songs = soup.find_all('h3')   # Song names are tagged as headers
    album_lyrics = {}
    for song in songs:
        song_name = re.findall(r'\d+\. (.*)', song.text)[0]   # Remove track number from song title
        logger.info("fetching lyrics for song " + song_name)
        album_lyrics[song_name] = get_song_lyrics(song)
    return album_lyrics


def get_song_lyrics(song):
    """Returns all text before the next song header tag
    """
    song_lyrics = ""
    for tag in song.next_siblings:
        if tag.name == 'h3':
            break
        elif isinstance(tag, str):
            song_lyrics += tag
    return song_lyrics


def get_album_url(band_name, album_name):
    def strip_name(name):
        return re.sub('[\W]+', '', name.lower())
    band_name_s = strip_name(band_name)
    album_name_s = strip_name(album_name)
    url = urljoin(BASEURL, 'lyrics/{}/{}.html'.format(band_name_s, album_name_s))
    return url


def main(src, dest):
    """Scrape Dark Lyrics for bands found in src directory and save updated bands to dest
    """
    # Get Metal-Archives info scraped by `metallum.py`
    logging.info("Getting band info from .json files in directory " + src)
    ma_filenames = [os.path.join(src, filename) for filename in os.listdir(src)][:10]
    t_bands = trange(len(ma_filenames), desc="looping over bands")
    for i in t_bands:
        try:
            filename = ma_filenames[i]
            band = json.load(open(filename, 'r'))
            logging.info("fetching albums by " + band['name'])
            t_albums = trange(
                len(band['albums']),
                desc="looping over albums by " + band['name'],
                leave=False
            )
            for j in t_albums:
                album = band['albums'][j]
                logging.info("fetching songs on '{}'".format(album['name']))
                album_url = get_album_url(band['name'], album['name'])
                try:
                    album_lyrics = get_album_lyrics(album_url)
                except ScrapingError:
                    logger.warning("page not found: " + album_url)
                except:
                    t_albums.close()
                    raise
                else:
                    logging.info("matching song lyrics with song data")
                    for song_idx, song in enumerate(album['songs']):
                        for song_name, song_lyrics in album_lyrics.items():
                            ma_name = unidecode(song['name'].lower())
                            dl_name = unidecode(song_name.lower())
                            # Update song with lyrics if song names match
                            if ma_name == dl_name:
                                logging.info("match found for " + song['name'])
                                new_info = dict(
                                    darklyrics=song_lyrics,
                                    darklyrics_url=album_url + "#" + str(song_idx + 1)
                                )
                                song.update(**new_info)
                                break
        except:
            t_bands.close()
            print()
            logging.error("error encountered while fetching lyrics", exc_info=1)
            sys.exit()
        output = os.path.join(dest, os.path.basename(filename))
        logging.info("saving updated band info to " + output)
        json.dump(band, open(output, 'w'), indent=4)
    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("src", help="source directory containing metal-archives bands")
    parser.add_argument("dest", help="destination directory for saving new data")
    parser.add_argument("--logfile", default=LOGFILE)
    args = parser.parse_args()
    logger = configure_logger(args.logfile)
    print("\nSCRAPING FROM {} WITH CRAWL DELAY {} SECONDS"
          .format(BASEURL, CRAWL_DELAY))
    print("PROGRESS WILL BE LOGGED IN {}\n".format(args.logfile))
    main(args.src, args.dest)
