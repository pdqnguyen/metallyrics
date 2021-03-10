"""
Script for fetching www.metal-archives.com id numbers of bands with more than 
the minimum number of reviews.
"""


import re
import os
import requests
import time
from argparse import ArgumentParser
import pandas as pd
from fake_useragent import UserAgent


CRAWLDELAY = 3  # Time between accessing pages; be nice, don't lower this number
BASEURL = 'http://www.metal-archives.com/review/ajax-list-browse/by/alpha/selection/'
ENDURL = '/json/1'
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
RESPONSELENGTH = 200   # Fetch in pages of 200 reviews at a time
MAXLENGTH = 12000      # Fetch up to 12000 reviews (highest of any letter (S) is >11,000 reviews)
MIN_REVIEWS = 0        # Minimum number of reviews for inclusion in table


def print_progress_bar(n, nmax, dt=None, length=20):
    """Call in a loop to create terminal progress bar
    """
    percent = ("{0:.1f}").format(100 * (n / nmax))
    filled = int(length * n // nmax)
    bar = '*' * filled + '-' * (length - filled)
    if dt is not None:
        dt_str = '({:.0f} seconds)'.format(dt)
    else:
        dt_str = ''
    print('\rscanning reviews: [{}] {}/{} entries {}'.format(bar, n, nmax, dt_str), end="\r")


def scrape_json(url, start=0, responselength=100):
    ua = UserAgent()
    payload = {'sEcho': 0, 'iDisplayStart': start, 'iDisplayLength': responselength}
    headers = {'User-Agent': ua.random}
    response = requests.get(url, params=payload, headers=headers).json()
    data = pd.DataFrame(response['aaData'])
    total = response['iTotalRecords']
    return data, total

def scrape_json_all(url, maxlength=MAXLENGTH, responselength=RESPONSELENGTH, crawldelay=CRAWLDELAY,):
    t0 = time.time()
    df_list = []
    for start in range(0, maxlength+1, responselength):
        data, total = scrape_json(url, start=start, responselength=responselength)
        print_progress_bar(start + responselength, total)
        df_list.append(pd.DataFrame(data))
        if len(data) < responselength:
            print_progress_bar(total, total, dt=time.time() - t0)
            print("")
            break
        time.sleep(crawldelay)
    df = pd.concat(df_list, axis=0, ignore_index=True)
    return df


def main(output, min_rev=MIN_REVIEWS):
    if not os.path.exists(output):
        os.makedirs(output)
    for letter in LETTERS:
        print('letter: ' + letter)
        t1 = time.time()
        url = BASEURL + letter + ENDURL
        df = scrape_json_all(url)
        df.columns = ['band_name', 'review_url', 'band_url', 'album_url', 'score', 'user_url', 'date']
        df['band_id'] = df['band_url'].apply(lambda x: re.search('(?<=\/)([0-9]+)', x).group(0))
        df = df[['band_name', 'band_id', 'band_url', 'review_url', 'album_url', 'score', 'user_url', 'date']]
        review_counts = df.groupby(['band_name', 'band_id']).apply(len)
        review_counts = review_counts[review_counts >= min_rev]
        if len(review_counts) > 0:
            band_id_most_reviews = review_counts.groupby(level=0).idxmax()
            band_ids = pd.DataFrame(list(band_id_most_reviews.values))
            band_ids.columns = ['name', 'id']
            filename = os.path.join(output, 'ids{}.csv'.format(letter))
            band_ids.to_csv(filename, index=False)
            print("{} unique bands found".format(len(band_ids)))
        else:
            print("letter {} has 0 bands above the minimum review count threshold of " + str(min_rev))
    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--min-rev", type=int, default=0)
    args = parser.parse_args()
    main(args.output, min_rev=args.min_rev)