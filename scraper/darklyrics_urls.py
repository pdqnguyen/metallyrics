import os
import json
import time
from argparse import ArgumentParser
from urllib.request import Request, urlopen
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from darklyrics_fast import configure_logger, strip_name, get_band_url, scrape_html,\
CRAWL_DELAY, BASEURL


LETTERS = 'abcdefghijklmnopqrstuvwxyz#'
LOGFILE = os.path.abspath('darklyrics_urls.log')


def get_darklyrics_links(letters=LETTERS, crawl_delay=CRAWL_DELAY):
    """
    Get all bands listed on DarkLyrics
    """
    out = {}
    t = tqdm(range(len(letters)), desc="looping over letters")
    for i in t:
        letter = letters[i]
        url = urljoin(BASEURL, letter + '.html')
        soup = scrape_html(url)
        artists = []
        for div in soup.find_all('div', attrs={'class': 'artists'}):
            artists.extend(div.find_all('a'))
        for a in artists:
            name = a.text
            if name not in out.keys():
                out[a.text] = urljoin(BASEURL, a['href'])
        time.sleep(CRAWL_DELAY)
    return out


def main(src, output, letters=None):
    """
    Match bands on DarkLyrics with Metal-Archives data fetched by metallum.py
    """
    src = os.path.abspath(src)
    logger.info(f"searching for .json files in directory {src}")
    src_files = [f for f in os.listdir(args.src) if '.json' in f]
    logger.info(f"{len(src_files)} .json files found")
    links = get_darklyrics_links(letters=letters)
    print(links)
    matched = {}
    for f in src_files:
        filename = os.path.join(src, f)
        band = json.load(open(filename, 'r'))
        band_name = strip_name(band['name'])
        band_url = get_band_url(band_name)
        if band_url in links.values():
            matched[filename] = band_url
    pd.Series(matched).to_csv(output, index_label='filename', header=['url'])
    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("src", help="source directory containing metal-archives bands")
    parser.add_argument("out", help="output csv file")
    parser.add_argument("--letters", default=LETTERS, help="search only bands starting with these letters")
    parser.add_argument("--logfile", default=LOGFILE)
    args = parser.parse_args()
    logger = configure_logger(args.logfile)
    print(f"\nSCRAPING FROM {BASEURL} WITH CRAWL DELAY {CRAWL_DELAY} SECONDS")
    print(f"PROGRESS WILL BE LOGGED IN {args.logfile}\n")
    main(args.src, args.out, letters=args.letters)
