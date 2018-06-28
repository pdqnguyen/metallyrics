import requests
import pandas as pd

LETTERS = 'NBR,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z'.split(',')
BASEURL = 'https://www.metal-archives.com/browse/ajax-letter/json/1/l/'
RESPONSELENGTH = 500

payload = {'sEcho': 0, 'iDisplayStart': 0, 'iDisplayLength': RESPONSELENGTH}
df_list = []
for letter in LETTERS[:3]:
    try:
        r = requests.get(BASEURL + letter, params=payload)
        js = r.json()
    except:
        pass
    data = js['aaData']
    df_list.append(pd.DataFrame(data))
df = pd.concat(df_list, axis=0, ignore_index=True)
band_names = df[0].str.split('/').apply(lambda x: x[-3])
df = pd.concat([df, band_names], axis=1, ignore_index=True)
df = df[[4, 1, 2, 0]]
df.columns = ['name', 'country', 'genre', 'link']
df.to_csv('bands.csv', index=False)