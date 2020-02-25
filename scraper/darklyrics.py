#! /usr/bin/env python
"""
Scrape all of Dark Lyrics for lyrics.
"""

import os
import json
import logging
import re
import time
from argparse import ArgumentParser
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urljoin

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
    logger.setLevel(logging.INFO)
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


def get_band_urls():
    band_urls = []
    t_letters = trange(len(LETTERS), desc="collecting band pages")
    for i in t_letters:
        letter = LETTERS[i]
        letter_url = urljoin(BASEURL, letter + ".html")
        logger.info("fetching page " + letter_url)
        try:
            soup = scrape_html(letter_url)
        except KeyboardInterrupt:
            t_letters.close()
            print()
            logger.error("operation aborted by user", exc_info=1)
            raise
        except ScrapingError:
            t_letters.close()
            print()
            logger.warning("failed to fetch page " + letter_url)
        except Exception:
            t_letters.close()
            print()
            raise
        else:
            logger.info("finding bands on page " + letter_url)
            artists = soup.find('div', attrs={'class': 'artists fr'})
            band_urls_letter = []
            if artists is not None:
                links = artists.find_all('a')
                if links is not None:
                    for link in links:
                        band_url = urljoin(BASEURL, link['href'])
                        band_urls_letter.append(band_url)
            logger.info("{} bands found starting with '{}'"
                        .format(len(band_urls_letter), letter))
            band_urls += band_urls_letter
    return band_urls


def get_band_lyrics(name, url):
    logger.info("fetching band page " + url)
    soup = scrape_html(url)
    if soup is None:
        return []
    logger.info("searching for albums")
    album_divs = soup.find_all('div', attrs={'class': 'album'})
    album_urls = []
    for i, album_div in enumerate(album_divs):
        album_href = album_div.find('a', href=True)['href']
        album_href = album_href.lstrip('.').rstrip('#1234567890')
        album_url = urljoin(BASEURL, album_href)
        album_urls.append(album_url)
    n_albums = len(album_urls)
    logger.info("{} albums found by {}".format(n_albums, name))
    band = {}
    t_albums = trange(n_albums, desc="collecting lyrics by " + name)
    for i in t_albums:
        album_url = album_urls[i]
        logger.info("searching for songs at " + album_url)
        album_name = re.findall(r'/(\w+)\.html', album_url)[0]
        band[album_name] = get_album_lyrics(album_url)
    return band


def get_album_lyrics(url):
    logger.info("fetching album page " + url)
    soup = scrape_html(url)
    if soup is None:
        return []
    logger.info("searching for songs")
    songs = soup.find_all('h3')
    album_lyrics = {}
    for song in songs:
        song_name = re.findall(r'\d+\. (.*)', song.text)[0]
        logger.info("fetching lyrics for song ".format(song_name))
        album_lyrics[song_name] = get_song_lyrics(song)
    return album_lyrics


def get_song_lyrics(song):
    song_lyrics = ""
    for tag in song.next_siblings:
        if tag.name == 'h3':
            break
        elif isinstance(tag, str):
            song_lyrics += tag
    return song_lyrics


def main(output):
    logger.info("collecting all band urls from " + BASEURL)
    band_urls = get_band_urls()
    n_bands = len(band_urls)
    logger.info("total of {} bands found".format(n_bands))
    logger.info("scraping all bands for lyrics")
    t_bands = trange(n_bands, desc="collecting lyrics")
    for j in t_bands:
        band_url = band_urls[j]
        logger.info("searching for albums at " + band_url)
        band_name = re.findall(r'/(\w+)\.html', band_url)[0]
        try:
            band = get_band_lyrics(band_name, band_url)
        except KeyboardInterrupt:
            t_bands.close()
            print()
            logger.error("operation aborted by user", exc_info=1)
            raise
        except ScrapingError:
            t_bands.close()
            print()
            logger.warning("failed to fetch page " + band_url)
        except Exception:
            t_bands.close()
            print()
            raise
        else:
            filename = os.path.join(output, band_name + '.json')
            logger.info("saving lyrics to ".format(filename))
            with open(os.path.join(filename), 'w') as f:
                json.dump(band, f)
    return


if __name__ == '__main__':
    logger = configure_logger()
    print("\nSCRAPING FROM {} WITH CRAWL DELAY {} SECONDS"
          .format(BASEURL, CRAWL_DELAY))
    print("PROGRESS WILL BE LOGGED IN {}\n".format(LOGFILE))
    parser = ArgumentParser()
    parser.add_argument("output", help="output directory")
    args = parser.parse_args()
    main(args.output)
