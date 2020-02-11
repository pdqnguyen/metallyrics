import re
import os
import time
import pandas as pd
from utils import scrape_json_all

BASEURL = 'http://www.metal-archives.com/review/ajax-list-browse/by/alpha/selection/'
ENDURL = '/json/1'
LETTERS = 'A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z'.split(',')
RESPONSELENGTH = 200
MAXLENGTH = 11000
CRAWLDELAY = 3
OUTDIR = 'ids-test/'

t0 = time.time()
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
for letter in LETTERS[25]:
    print('Letter: ' + letter)
    t1 = time.time()
    url = BASEURL + letter + ENDURL
    df = scrape_json_all(url, crawldelay=CRAWLDELAY, start=0, maxlength=MAXLENGTH, responselength=RESPONSELENGTH) 
    df.columns = ['band_name', 'review_url', 'band_url', 'album_url', 'score', 'user_url', 'date']
    df['band_id'] = df['band_url'].apply(lambda x: re.search('(?<=\/)([0-9]+)', x).group(0))
    df = df[['band_name', 'band_id', 'band_url', 'review_url', 'album_url', 'score', 'user_url', 'date']]
    review_counts = df.groupby(['band_name', 'band_id']).apply(len)
    band_id_most_reviews = review_counts.groupby(level=0).idxmax()
    band_ids = pd.DataFrame(list(band_id_most_reviews.values))
    band_ids.columns = ['name', 'id']
    band_ids.to_csv(OUTDIR + 'ids' + letter + '.csv', index=False)
    print('Letter {} complete: {:.0f} seconds'.format(letter, time.time() - t1))
print('Scraping complete: {:.0f} seconds'.format(time.time() - t0))