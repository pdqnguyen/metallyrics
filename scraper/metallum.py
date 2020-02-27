#! /usr/bin/env python
"""
Scrape Metal Archives for band/album/song information.
"""

import os
import json
import time
import logging
import sys
from argparse import ArgumentParser
from urllib.parse import urljoin, quote_plus
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import trange


USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'
CRAWL_DELAY = 3  # Time between accessing pages; be nice, don't lower this number
BASEURL = 'https://www.metal-archives.com'
LOGFILE = os.path.abspath('scraper_metallum.log')


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
    file_handler = logging.FileHandler(logfile, mode='w', encoding='utf-8')
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


def get_band_info(url):
    """Get basic info from band page
    """
    soup = scrape_html(url)
    raw_info = soup.find('div', attrs={'id': 'band_stats'})
    keys = raw_info.find_all('dt')
    vals = raw_info.find_all('dd')
    try:
        info = {key.text.replace(':', ''): val.text for key, val in zip(*(keys, vals))}
    except (AttributeError, ValueError):
        logger.error("error while parsing band info")
    else:
        return info


def get_discography(url, full_length_only=False):
    """Get album info and links from discography table
    """
    def parse_review_str(s):
        try:
            num, avg = s.split()
        except ValueError:
            return 0, None
        else:
            num = int(num)
            avg = int(avg.strip('()%'))
            return num, avg
    soup = scrape_html(url)
    table = soup.find('table', attrs={'class': 'display discog'}).tbody
    out = []
    for tr in table.find_all('tr'):
        cells = tr.find_all('td')
        if len(cells) == 4:
            name_c, type_c, year_c, reviews_c = cells
            try:
                name = name_c.text.strip()
                type_ = type_c.text.strip()
                year = int(year_c.text.strip())
                reviews = reviews_c.text.strip()
                album_url = name_c.a['href']
            except (AttributeError, TypeError):
                logger.warning("error while parsing table from " + url)
            else:
                # Skip non-full-length albums if desired
                if full_length_only and type_ != 'Full-length':
                    continue
                try:
                    reviews_url = reviews_c.a['href']
                except (AttributeError, TypeError):
                    reviews_url = None
                review_num, review_avg = parse_review_str(reviews)
                info = dict(
                    name=name,
                    type=type_,
                    year=year,
                    review_num=review_num,
                    review_avg=review_avg,
                    url=album_url,
                    reviews_url=reviews_url
                )
                out.append(info)
    return out


def get_reviews(url):
    """Get review titles, content, and dates for an album
    """
    soup = scrape_html(url)
    out = []
    for box in soup.find_all(attrs={'class': 'reviewBox'}):
        title = box.find(attrs={'class': 'reviewTitle'})
        content = box.find(attrs={'class': 'reviewContent'})
        date = box.find('a', attrs={'class': 'profileMenu'})
        try:
            review = dict(
                title=title.text.strip(),
                content=content.text.strip(),
                date=date.next.next.strip(', \n')
            )
        except AttributeError:
            logger.warning("error while parsing review from " + url)
        else:
            out.append(review)
    return out


def get_tracklist(url):
    """Get song info and links from album page
    """
    soup = scrape_html(url)
    table = soup.find('table', attrs={'class': 'display table_lyrics'}).tbody
    out = []
    for tr in table.find_all('tr'):
        cells = tr.find_all('td')
        if len(cells) == 4:
            try:
                name = cells[1].text.strip()
                length = cells[2].text.strip()
                lyrics = cells[3].text.strip()
            except AttributeError:
                logger.warning("error while parsing table from " + url)
            else:
                if lyrics != '':
                    link = tr.find('a')
                    id_ = link['name']
                    song_url = urljoin(BASEURL, 'release/ajax-view-lyrics/id/' + id_)
                else:
                    song_url = None
                info = dict(
                    name=name,
                    length=length,
                    url=song_url
                )
                out.append(info)
    return out


def get_lyrics(url):
    """Get lyrics for a song
    """
    soup = scrape_html(url)
    return soup.text.strip()


def get_album(name, url, omit_lyrics=False):
    """Get album info and song lyrics
    """
    # Get basic song info
    try:
        songs = get_tracklist(url)
    except ScrapingError:
        logger.warning("error while loading album page: " + url, exc_info=1)
        songs = []
    # Get lyrics for each song
    if not omit_lyrics:
        for k in trange(
                len(songs),
                desc='fetching songs from "{}"'.format(name),
                # bar_format="{desc} {percentage:3.0f}%|{bar:20}{r_bar}|",
                leave=False
        ):
            song = songs[k]
            if song['url'] is not None:
                logger.info("fetching song " + song['name'])
                try:
                    song['lyrics'] = get_lyrics(song['url'])
                except ScrapingError:
                    logger.warning("error while loading song page: " + song['url'], exc_info=1)
                    song['lyrics'] = None
            else:
                logger.info("no lyrics found for song: " + song['name'])
                song['lyrics'] = None
    out = dict(
        name=name,
        url=url,
        songs=songs,
        song_names=[song['name'] for song in songs],
    )
    return out


def get_band(name, url, full_length_only=False, omit_lyrics=False):
    """Get band info and albums/songs
    """
    t0 = time.time()
    # Get basic band information
    try:
        info = get_band_info(url)
    except ScrapingError:
        logger.error("error encoutered while loading band page: " + url, exc_info=1)
        raise
    # Get basic album info
    id_ = os.path.split(url)[-1]
    disco_url = urljoin(BASEURL, 'band/discography/id/' + id_ + '/tab/all')
    try:
        albums = get_discography(disco_url, full_length_only=full_length_only)
    except ScrapingError:
        logger.warning("error while loading discography page: " + url, exc_info=1)
        albums = []
    # Get reviews and lyrics for each album
    for j in trange(
            len(albums),
            desc='fetching albums by {}'.format(name),
            # bar_format="{desc} {percentage:3.0f}%|{bar:20}{r_bar}|",
            leave=False
    ):
        album = albums[j]
        logger.info("fetching album " + album['name'])
        if album['reviews_url'] is not None:
            try:
                album['reviews'] = get_reviews(album['reviews_url'])
            except ScrapingError:
                logger.warning("error while loading reviews page: " + album['reviews_url'], exc_info=1)
                album['reviews'] = []
        else:
            album['reviews'] = []
        album_songs = get_album(album['name'], album['url'], omit_lyrics=omit_lyrics)
        album.update(album_songs)
    out = dict(
        name=name,
        id=id_,
        url=url,
        albums=albums,
        album_names=[album['name'] for album in albums],
        info=info
    )
    dt = time.time() - t0
    logger.info("finished fetching {}".format(name, dt / 60.))
    return out


def main(filename, output, full_length_only=False, omit_lyrics=False):
    """Download and save band data for every band/id pair in csv.
    
    Parameters
    ----------
    filename : str
        Table of band names and id numbers collected by metallum_ids.py
    output : str
        Directory for .json output files
    full_length_only : {True, False}
    omit_lyrics : {True, False
    """
    t0 = time.time()
    if not os.path.exists(output):
        os.makedirs(output)
    band_df = pd.read_csv(filename)
    logger.info("fetching bands from {}".format(filename))
    t = trange(
        len(band_df),
        desc="collecting artist data",
        # bar_format="{desc} {percentage:3.0f}%|{bar:20}{r_bar}|",
    )
    for i in t:
        row = band_df.iloc[i]
        name = row['name']
        id_ = str(row['id'])
        url = urljoin(BASEURL, 'bands/' + quote_plus(name) + '/' + id_)
        logger.info("fetching band " + name)
        try:
            band = get_band(name, url,
                            full_length_only=full_length_only,
                            omit_lyrics=omit_lyrics)
        except KeyboardInterrupt:
            t.close()
            print()
            logger.error("scraping aborted by user", exc_info=1)
            sys.exit()
        except ScrapingError:
            logger.warning("failed to fetch " + name)
        else:
            basename = '_'.join(url.rstrip('/').split('/')[-2:]) + '.json'
            filename = os.path.join(output, basename)
            with open(filename, 'w') as f:
                json.dump(band, f, indent=4)
    dt = time.time() - t0
    logger.info("finished fetching all bands: {:.0f} minutes".format(dt / 60.))
    return


if __name__ == '__main__':
    logger = configure_logger()
    print("\nSCRAPING FROM {} WITH CRAWL DELAY {} SECONDS"
          .format(BASEURL, CRAWL_DELAY))
    print("PROGRESS WILL BE LOGGED IN {}\n".format(LOGFILE))
    parser = ArgumentParser()
    parser.add_argument("ids")
    parser.add_argument("outdir")
    parser.add_argument("--full-length-only", action="store_true", default=False)
    parser.add_argument("--omit-lyrics", action="store_true", default=False)
    args = parser.parse_args()
    main(args.ids, args.outdir,
         full_length_only=args.full_length_only, omit_lyrics=args.omit_lyrics)
