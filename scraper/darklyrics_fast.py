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

from unidecode import unidecode
from bs4 import BeautifulSoup
from tqdm import tqdm
from fake_useragent import UserAgent


PROXIES_URL = 'https://free-proxy-list.net/'
USE_PROXIES = False
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


def scrape_html(url, proxy=None, crawl_delay=CRAWL_DELAY, scrape_timout=SCRAPE_TIMEOUT):
    try:
        if proxy is not None:
            req = Request(url=url)
            req.set_proxy(proxy['ip'] + ':' + proxy['port'], 'http')
        else:
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


def scrape_html_with_proxies(url, retry=5):
    global proxies, proxy
    for i in range(retry):
        try:
            soup = scrape_html(url, proxy=proxy)
        except (URLError, ConnectionError, TimeoutError, HTTPException):
            logger.warning(f"connection error while fetching {url}")
            if i < retry:
                proxy = proxies.pop(0)
                logging.warning(f"retrying with new proxy ({i + 1}/{retry})")
        except ScrapingError:
            logger.warning(f"page not found: {url}")
            if i < retry:
                proxy = proxies.pop(0)
                logging.warning(f"retrying with new proxy ({i + 1}/{retry})")
        else:
            return soup
    raise ScrapingError(f"failed to fetch {url} after {retry} attempts")


def get_proxies(proxies_url=PROXIES_URL):
    ua = UserAgent()
    req = Request(proxies_url)
    req.add_header('User-Agent', ua.random)
    page = urlopen(req).read().decode('utf8')
    soup = BeautifulSoup(page, 'html.parser')
    proxies_table = soup.find(id='fpl-list')
    out = []
    for row in proxies_table.tbody.find_all('tr'):
        cells = row.find_all('td')
        out.append(dict(
            ip=cells[0].string,
            port=cells[1].string
        ))
    random.shuffle(out)
    return out


def get_random_proxy(proxies_list):
    idx = random.choice(range(len(proxies_list)))
    return idx, proxies_list[idx]


def band_exists(name, use_proxies=USE_PROXIES):
    url = get_band_url(name)
    out = False
    if use_proxies:
        global proxies, proxy
        try:
            scrape_html_with_proxies(url)
        except ScrapingError:
            pass
        else:
            out = True
    else:
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


def get_band_lyrics(band, use_proxies=USE_PROXIES):
    band_name = strip_name(band['name'])
    logger.info(f"fetching lyrics by {band_name}")
    if band_exists(band_name, use_proxies=use_proxies):
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
            if use_proxies:
                global proxies, proxy
                if len(proxies) < 10:
                    proxies = get_proxies()
                    proxy = proxies.pop(0)
            try:
                album_lyrics = get_album_lyrics(album_url, use_proxies=use_proxies)
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


def get_album_lyrics(url, use_proxies=USE_PROXIES):
    """Fetch all lyrics for an album
    """
    logger.info(f"fetching lyrics from {url}")
    if use_proxies:
        global proxies, proxy
        soup = scrape_html_with_proxies(url)
    else:
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


def main(src, dest, use_proxies=USE_PROXIES):
    """Scrape Dark Lyrics for bands found in src directory and save updated bands to dest
    """
    global proxies, proxy
    # Get Metal-Archives info scraped by `metallum.py`
    logger.info(f"searching for .json files in directory {src}")
    ma_filenames = []
    for f in os.listdir(src):
        if '.json' in f:
            filename = os.path.join(src, f)
            output = os.path.join(dest, os.path.basename(filename))
            if os.path.exists(output):
                file_has_lyrics = check_file_for_lyrics(output)
                if file_has_lyrics:
                    logger.info("file already exists and contains lyrics: " + output)
                else:
                    logger.info("file exists; lyrics will be added to it:" + output)
                    ma_filenames.append(filename)
            else:
                ma_filenames.append(filename)
    t = tqdm(range(len(ma_filenames)), desc="looping over bands")
    for i in t:
        filename = ma_filenames[i]
        output = os.path.join(dest, os.path.basename(filename))
        try:
            band = json.load(open(filename, 'r'))
            band.update(get_band_lyrics(band, use_proxies=use_proxies))
        except KeyboardInterrupt:
            t.close()
            print()
            logger.error("scraping aborted by user", exc_info=True)
            sys.exit()
        except ScrapingError:
            logger.warning("failed to fetch " + name)
        else:
            logger.info(f"saving updated band info to {output}")
            json.dump(band, open(output, 'w'), indent=4)
    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("src", help="source directory containing metal-archives bands")
    parser.add_argument("--dest",
                        help="destination directory for saving new data; " +
                             "overwrite src if not given")
    parser.add_argument("--use-proxies", action="store_true", default=USE_PROXIES,
                        help="use proxies to rotate IP address while scraping")
    parser.add_argument("--logfile", default=LOGFILE)
    args = parser.parse_args()
    logger = configure_logger(args.logfile)
    print(f"\nSCRAPING FROM {BASEURL} WITH CRAWL DELAY {CRAWL_DELAY} SECONDS")
    print(f"PROGRESS WILL BE LOGGED IN {args.logfile}\n")
    if args.dest:
        dest = args.dest
    else:
        dest = args.src
    if args.use_proxies:
        proxies = get_proxies()
        proxy = proxies.pop(0)
    else:
        proxies = None
        proxy = None
    main(args.src, dest, use_proxies=args.use_proxies)
