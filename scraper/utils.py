import requests
import pandas as pd
import time
from urllib.request import urlopen, HTTPError, Request
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords


USER_AGENT = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'


def scrape_json(url, start=0, responselength=100, user_agent=USER_AGENT):
    payload = {'sEcho': 0, 'iDisplayStart': start, 'iDisplayLength': responselength}
    headers = {'User-Agent': user_agent}
    r = requests.get(url, params=payload, headers=headers)
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

def scrape_html(url, user_agent=USER_AGENT):
    headers = {'User-Agent': user_agent}
    req = Request(url=url, headers=headers)
    try:
        page = urlopen(req).read()
        soup = BeautifulSoup(page, 'html.parser')
    except HTTPError:
        print('url not found: ' + url)
        soup = None
    return soup
