import re
import pandas as pd
from nltk.corpus import stopwords

def strip_lyrics(lyrics):
    text = ' '.join(lyrics)
    words = re.split(' |,|\.', text)
    out = [word.lower() for word in words if word != '']
    return out

def word_count(a):
    words = []
    for word in a:
        if any(char.isalpha() for char in word):
            try:
                word = re.match('[a-zA-Z0-9]+', word).group(0)
            except AttributeError:
                pass
            else:
                words.append(word)
    keys = sorted(set(words) - set(stopwords.words('english')))
    return {key: words.count(key) for key in keys if '\'re' not in key}

def band_word_count(band_lyrics):
    if len(band_lyrics) == 0:
        return pd.DataFrame()
    df_list = []
    for album_lyrics in band_lyrics.values():
        for song, lyrics in album_lyrics.items():
            word_count_dict = word_count(strip_lyrics(lyrics))
            song_df = pd.DataFrame.from_dict(word_count_dict, orient='index')
            if len(song_df) != 0:
                song_name = song[song.index('.')+2:]
                song_df.columns = [song_name]
                df_list.append(song_df)
    df = pd.concat(df_list, axis=1).fillna(0)
    return df