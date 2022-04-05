"""
Populate the output of metallum.py with lyrics fetched from Dark Lyrics.
"""
import os
import json
import logging
import random
import re
import sys
import time
from argparse import ArgumentParser
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from http.client import HTTPException
from socket import timeout

import pandas as pd
from unidecode import unidecode
from bs4 import BeautifulSoup
from tqdm import tqdm
from fake_useragent import UserAgent


SCRAPE_TIMEOUT = 5
CRAWL_DELAY = 10  # Time between accessing pages; be nice, don't lower this number
BASEURL = 'http://www.darklyrics.com'
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


def scrape_html(url, crawl_delay=CRAWL_DELAY, scrape_timout=SCRAPE_TIMEOUT):
    try:
        ua = UserAgent()
        req = Request(url=url, headers={'User-Agent': ua.random})
        page = urlopen(req, timeout=scrape_timout).read()
        soup = BeautifulSoup(page, 'html.parser')
        assert len(soup.text) > 0
    except AssertionError:
        raise URLError(f"empty page {url}")
    except (HTTPError, ValueError, timeout):
        raise ScrapingError(f"invalid url {url}")
    finally:
        time.sleep(crawl_delay)
    return soup


def band_exists(name, url=None):
    if url is None:
        url = get_band_url(name)
    out = False
    try:
        scrape_html(url)
    except ScrapingError:
        pass
    else:
        out = True
    return out


def check_file_for_lyrics(filename):
    band = json.load(open(filename, 'r'))
    out = False
    for album in band['albums']:
        for song in album['songs']:
            if 'darklyrics' in song.keys():
                if len(song.get('darklyrics', '')) > 0:
                    out = True
    return out


def get_band_url(band_name):
    band_name_s = strip_name(band_name)
    letter = band_name_s[0]
    url = urljoin(BASEURL, f"{letter}/{band_name_s}.html")
    return url


def get_album_url(band_name, album_name):
    band_name_s = strip_name(band_name)
    album_name_s = strip_name(album_name)
    url = urljoin(BASEURL, f"lyrics/{band_name_s}/{album_name_s}.html")
    return url


def strip_name(name):
    # Remove non-alphabetic characters
    stripped = re.sub(r'[^a-zA-Z]', r'', name).lower()
    return stripped


def get_band_lyrics(band, url=None):
    band_name = strip_name(band['name'])
    logger.info(f"fetching lyrics by {band_name}")
    if url is None:
        # No band url provided, we have to check if the band has a page on DL
        this_band_exists = band_exists(band_name)
    else:
        # Band url provided, assume they exist on DL
        this_band_exists = True
    if this_band_exists:
        logging.info(f"band {band_name} exists on {BASEURL}")
        t_albums = tqdm(
            range(len(band['albums'])),
            desc=f"looping over albums by {band_name}",
            leave=False
        )
        for j in t_albums:
            album = band['albums'][j]
            album_name = strip_name(album['name'])
            logger.info(f"fetching album '{album_name}'")
            album_url = get_album_url(band_name, album_name)
            try:
                album_lyrics = get_album_lyrics(album_url)
            except ScrapingError:
                logger.warning(f"page not found: {album_url}")
            except:
                t_albums.close()
                raise
            else:
                logger.info("matching song lyrics with song data")
                for song_idx, song in enumerate(album['songs']):
                    for song_name, song_lyrics in album_lyrics.items():
                        ma_name = unidecode(song['name'].lower())
                        dl_name = unidecode(song_name.lower())
                        # Update song with lyrics if song names match
                        if ma_name == dl_name:
                            logger.info(f"match found for {dl_name}")
                            new_info = dict(
                                darklyrics=song_lyrics,
                                darklyrics_url=f"{album_url}#{song_idx + 1}"
                            )
                            song.update(**new_info)
                            break
    else:
        logger.warning(f"band {band_name} not found on {BASEURL}")
    return band


def get_album_lyrics(url):
    """Fetch all lyrics for an album
    """
    logger.info(f"fetching lyrics from {url}")
    soup = scrape_html(url)
    songs = soup.find_all('h3')   # Song names are tagged as headers
    album_lyrics = {}
    for song in songs:
        song_name = re.findall(r'\d+\. (.*)', song.text)[0]   # Remove track number from song title
        logger.info(f"fetching lyrics for song '{strip_name(song_name)}'")
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


def get_urls_from_csv(filename):
    """Get DarkLyrics band urls generated from darklyrics_urls.py
    """
    df = pd.read_csv(filename)
    filenames = df['filename'].values
    urls = df['url'].values
    return dict(zip(filenames, urls))


def main(src, dest=None):
    """Scrape Dark Lyrics for bands found in src directory and save updated bands to dest
    """
    # Get Metal-Archives info scraped by `metallum.py`
    if '.csv' in src:
        darklyrics_urls = get_urls_from_csv(src)
        ma_filenames = list(darklyrics_urls.keys())
    else:
        ma_filenames = []
        for f in os.listdir(src):
            if '.json' in f:
                ma_filenames.append(os.path.join(src, f))
        darklyrics_urls = {}
    outputs = {}
    for filename in ma_filenames:
        if dest is not None:
            output = os.path.join(dest, os.path.basename(filename))
        else:
            output = os.path.abspath(filename)
        if os.path.exists(output):
            file_has_lyrics = check_file_for_lyrics(output)
            if file_has_lyrics:
                logger.info(f"file already exists and contains lyrics: {output}")
            else:
                logger.info(f"file exists; lyrics will be added to it: {output}")
                outputs[filename] = output
        else:
            outputs[filename] = output
    t = tqdm(range(len(outputs)), desc="looping over bands")
    for i in t:
        filename, output = list(outputs.items())[i]
        url = darklyrics_urls.get(filename, None)
        try:
            band = json.load(open(filename, 'r'))
            band.update(get_band_lyrics(band, url=url))
        except KeyboardInterrupt:
            t.close()
            print()
            logger.error("scraping aborted by user", exc_info=True)
            sys.exit()
        except ScrapingError:
            logger.warning(f"failed to fetch {filename}")
        else:
            logger.info(f"saving updated band info to {output}")
            json.dump(band, open(output, 'w'), indent=4)
    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("src", help="source directory containing metal-archives bands, "
                                      "or a csv file containing paths to metal-archives bands and darklyrics urls")
    parser.add_argument("--dest",
                        help="destination directory for saving new data; " +
                             "overwrite src if not given")
    parser.add_argument("--logfile", default=LOGFILE)
    args = parser.parse_args()
    logger = configure_logger(args.logfile)
    print(f"\nSCRAPING FROM {BASEURL} WITH CRAWL DELAY {CRAWL_DELAY} SECONDS")
    print(f"PROGRESS WILL BE LOGGED IN {args.logfile}\n")
    if args.dest:
        dest = args.dest
        if not os.path.exists(dest):
            os.makedirs(dest)
    elif '.csv' in args.src:
        dest = None
    else:
        dest = args.src
    main(args.src, dest)
