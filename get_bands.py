import requests
import pandas as pd
import re
import os
import time
from urllib.parse import unquote
from utils import scrape_json_all

LETTERS = 'A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z'.split(',')
BASEURL = 'https://www.metal-archives.com/browse/ajax-letter/json/1/l/'
RESPONSELENGTH = 500
MAXLENGTH = 14000
CRAWLDELAY = 3

t0 = time.time()
if not os.path.exists('bands/'):
    os.makedirs('bands/')
for letter in LETTERS[2]:
    print('Letter: ' + letter)
    t1 = time.time()
    url = BASEURL + letter
    df = scrape_json_all(url, crawldelay=CRAWLDELAY, start=0, maxlength=MAXLENGTH, responselength=RESPONSELENGTH)    
    band_urls = df[0].apply(lambda x: re.search('(?<=\')(.*)(?=\')', x).group(0))
    band_names = band_urls.apply(lambda x: re.search('(?<=(bands\/))(.*)(?=\/)', x).group(0))
    band_names = band_names.apply(lambda x: unquote(x).replace('_', ' '))
    band_ids = band_urls.apply(lambda x: re.search('(?<=(\/))([0-9]+)', x).group(0))
    #band_names = df[0].str.split('/').apply(lambda x: x[-3])
    df = pd.concat([df[[1, 2]], band_names, band_ids, band_urls], axis=1, ignore_index=True)
    df = df[[2, 3, 0, 1, 4]]
    df.columns = ['name', 'id', 'country', 'genre', 'url']
    #df.url = df.url.apply(lambda x: re.search('(?<=\')(.*)(?=\')', x).group(0))
    df.to_csv('bands/bands' + letter + '.csv', index=False)
    print('Letter {} complete: {:.0f} seconds'.format(letter, time.time() - t1))
print('Scraping complete: {:.0f} seconds'.format(time.time() - t0))