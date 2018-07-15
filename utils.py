import requests
import pandas as pd
import time
from urllib.request import urlopen, HTTPError
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def scrape_json(url, start=0, responselength=100):
    payload = {'sEcho': 0, 'iDisplayStart': start, 'iDisplayLength': responselength}
    r = requests.get(url, params=payload)
    data = pd.DataFrame(r.json()['aaData'])
    return data

def scrape_json_all(url, crawldelay=1, start=0, maxlength=100, responselength=100):
    df_list = []
    for new_start in range(0, maxlength+1, responselength):
        print(new_start)
        data = scrape_json(url, start=new_start, responselength=responselength)
        df_list.append(pd.DataFrame(data))
        if len(data) < responselength:
            break
        time.sleep(crawldelay)
    df = pd.concat(df_list, axis=0, ignore_index=True)
    return df

def scrape_html(url):
    try:
        with urlopen(url) as page:
            soup = BeautifulSoup(page, 'html.parser')
    except HTTPError:
        print('url not found: ' + url)
        soup = None
    return soup

def get_genres(genre_string):
    pattern = '[a-zA-Z\-]+'
    genres = [match.lower() for match in re.findall(pattern, genre_string)]
    genres = sorted(set(genres))
    if 'metal' in genres:
        genres.remove('metal')
    return genres