
# Song lyrics - Part 1: Overview

This is the first of several notebooks examining the song lyrics dataset. This notebook explores some basic properties of the dataset, while also tidying up the data for analysis in the following notebooks.

### Imports


```python
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
plt.style.use('seaborn')
import seaborn as sns
sns.set(font_scale=2)
from nltk.corpus import words as nltk_words
from scipy.stats import linregress

from adjustText import adjust_text

from nlp import tokenize
```

### Data

The dataset used here is the table of artist/album/song info and lyrics for every song in the core dataset.


```python
df = pd.read_csv('songs.csv', low_memory=False)
df = df[~df.song_darklyrics.isnull()]
df = df[df.song_darklyrics.str.strip().apply(len) > 0]
print(df.columns)
```

    Index(['band_name', 'band_id', 'band_url', 'band_country_of_origin',
           'band_location', 'band_status', 'band_formed_in', 'band_genre',
           'band_lyrical_themes', 'band_last_label', 'band_years_active',
           'album_name', 'album_type', 'album_year', 'album_review_num',
           'album_review_avg', 'album_url', 'album_reviews_url', 'song_name',
           'song_length', 'song_url', 'song_darklyrics', 'song_darklyrics_url',
           'band_current_label'],
          dtype='object')
    

### Reduced dataset

For lyrical analyses the data is reduced to just a column of lyrics (which will become the feature vector upon some transformation to a quantitative representation) for each song and columns for the most popular genres (the target/label vectors). These are the genres that appear at least once in isolation, i.e. not accompanied by any other genre, and that appear in some minimum percentage of songs. For example, the "black" metal label can appear on bands with or without other genres, but a label like "atmospheric" never appears on its own despite being fairly popular, usually because it is more of an adjective to denote subgenres like atmospheric black metal; thus "black" is included in the reduced label space but "atmospheric" is not. This reduces the genres to a more manageable set: five genres if the minimum occurrence requirement is set to 10%, and thirteen if set to 1%.

A five-genre set would be easier to handle but leaves quite a few holes in the label space, because doom metal, metalcore, folk metal, and many other fairly popular genres are being omitted that may not be covered by any of the five labels. The larger label set covers just about all the most important genres, but because eight of them occur in fewer than 10% of all songs, they will force greater class imbalance which will adversely affect attempts at applying binary classification models later on. For the sake of comparison, both reduced datasets are saved here, but the rest of this exploratory analysis only looks at the 1% dataset, while the 10% dataset is reserved for modeling. Each dataset is saved in its raw form and in a truncated (ML-ready) form containing only the lyrics and genre columns.


```python
def process_genre(genre):
    # Find words (including hyphenated words) not in parentheses
    out = re.findall('[\w\-]+(?![^(]*\))', genre.lower())
    out = [s for s in out if s != 'metal']
    return out


song_genres = df.band_genre.apply(process_genre)
genres = sorted(set(song_genres.sum()))
genre_cols = [f'genre_{genre}' for genre in genres]
for genre, col in zip(genres, genre_cols):
    df[col] = song_genres.apply(lambda x: int(genre in x))
```


```python
def get_top_genres(data, min_pct):
    isolated = (data.sum(axis=1) == 1)
    isolated_cols = sorted(set(data[isolated].idxmax(axis=1)))
    top_cols = [col for col in isolated_cols if data[col][isolated].mean() >= min_pct]
    top_genres = [re.sub(r"^genre\_", "", col) for col in top_cols]
    return top_genres
```


```python
top_genres_10pct = get_top_genres(df[genre_cols], 0.1)
print(top_genres_10pct)
df_rr = df.copy()
drop_cols = [col for col in df.columns if ('genre_' in col) and (re.sub(r"^genre\_", "", col) not in top_genres_10pct)]
df_rr.drop(drop_cols, axis=1, inplace=True)
df_rr.to_csv('songs-10pct.csv', index=False)

# Only lyrics and genre are relevant for ML later
df_rr_ml = pd.DataFrame(index=range(df.shape[0]), columns=['lyrics'] + top_genres_10pct)
df_rr_ml['lyrics'] = df['song_darklyrics'].reset_index(drop=True)
df_rr_ml[top_genres_10pct] = df[[f"genre_{genre}" for genre in top_genres_10pct]].reset_index(drop=True)
df_rr_ml.to_csv('songs-ml-10pct.csv', index=False)
```

    ['black', 'death', 'heavy', 'power', 'thrash']
    


```python
top_genres_1pct = get_top_genres(df[genre_cols], 0.01)
print(top_genres_1pct)
df_r = df.copy()
drop_cols = [col for col in df.columns if ('genre_' in col) and (re.sub(r"^genre\_", "", col) not in top_genres_1pct)]
df_r.drop(drop_cols, axis=1, inplace=True)
df_r.to_csv('songs-1pct.csv', index=False)

# Only lyrics and genre are relevant for ML later
df_r_ml = pd.DataFrame(index=range(df.shape[0]), columns=['lyrics'] + top_genres_1pct)
df_r_ml['lyrics'] = df['song_darklyrics'].reset_index(drop=True)
df_r_ml[top_genres_1pct] = df[[f"genre_{genre}" for genre in top_genres_1pct]].reset_index(drop=True)
df_r_ml.to_csv('songs-ml-1pct.csv', index=False)
```

    ['black', 'death', 'deathcore', 'doom', 'folk', 'gothic', 'grindcore', 'heavy', 'metalcore', 'power', 'progressive', 'symphonic', 'thrash']
    

# Basic lyrical properties

This section compares looks at word counts and unique word counts, in absolute counts as well as counts per minute, between different songs, albums, bands, and genres. [Part 3](./lyrics2.ipynb) dives much deeper into evaluating lyrical complexity using various lexical diversity measures from the literature.

Song lyrics are tokenized using a custom `tokenize()` function in `nlp.py`.

# Word counts by song


```python
song_tokens = df_r.song_darklyrics.apply(tokenize)
song_word_counts = song_tokens.apply(len)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
song_word_counts.hist(bins=np.logspace(1, 3, 30), ax=ax)
ax.set_xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel("word count")
ax.set_ylabel("number of songs")
ax.set_title("Words per song")
plt.show()
```


![png](lyrics0_files/lyrics0_12_0.png)



```python
def to_seconds(data):
    """Convert a time string (MM:ss or HH:MM:ss) to seconds
    """
    out = pd.Series(index=data.index, dtype=int)
    for i, x in data.song_length.items():
        if isinstance(x, str):
            xs = x.split(':')
            if len(xs) < 3:
                xs = [0] + xs
            seconds = int(xs[0]) * 3600 + int(xs[1]) * 60 + int(xs[2])
        else:
            seconds = 0
        out[i] = seconds
    return out
```


```python
song_seconds = to_seconds(df_r)
song_words_per_second = song_word_counts / song_seconds
song_words_per_second[song_words_per_second == np.inf] = 0

df_r_songs = df_r[['band_name', 'album_name', 'song_name', 'band_genre']].copy()
df_r_songs['song_word_count'] = song_word_counts
df_r_songs['song_seconds'] = song_seconds
df_r_songs['song_words_per_second'] = song_words_per_second
```

### Songs with highest word counts

The honor of highest word count in a single song goes to the [Bal-Sagoth's "The Obsidian Crown Unbound"](https://youtu.be/xizMG4nI2dk) at over two thousand words. However, most of those words are not sung in the actual song: Bal-Sagoth lyrics typically include the massive collection of narrative text that accompanies their songs. Although the lyrics they sing are still plentiful, there are nowhere near two thousand words spoken in the six-minute symphonic black metal track.

This makes the forty-minute prog metal epic [Crimson by Edge of Sanity](https://youtu.be/St6lJaiHYIc) a better contender for most verbose song. Still, such a claim might be challenged by the fact that the digital edition of the album, which a listener would find on Spotify for instance, divides the single-track album into eight parts. That said, DarkLyrics keeps the original one-track format.

At third place is another multi-part song, [Mirror of Souls](https://youtu.be/y6n1kMsLbc8) by the Christian progressive/power metal group Theocracy. This is less contentious since the official track listing considers this a single track.


```python
df_r_songs.sort_values('song_word_count')[:-10:-1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band_name</th>
      <th>album_name</th>
      <th>song_name</th>
      <th>band_genre</th>
      <th>song_word_count</th>
      <th>song_seconds</th>
      <th>song_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11862</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>The Obsidian Crown Unbound</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>2391</td>
      <td>358</td>
      <td>6.678771</td>
    </tr>
    <tr>
      <th>35657</th>
      <td>Edge of Sanity</td>
      <td>Crimson</td>
      <td>Crimson</td>
      <td>Progressive Death Metal</td>
      <td>2019</td>
      <td>2400</td>
      <td>0.841250</td>
    </tr>
    <tr>
      <th>100430</th>
      <td>Theocracy</td>
      <td>Mirror of Souls</td>
      <td>Mirror of Souls</td>
      <td>Epic Progressive Power Metal</td>
      <td>1603</td>
      <td>1346</td>
      <td>1.190936</td>
    </tr>
    <tr>
      <th>11822</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>To Dethrone the Witch-Queen of Mytos K'unn (Th...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>1413</td>
      <td>405</td>
      <td>3.488889</td>
    </tr>
    <tr>
      <th>11866</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Unfettering the Hoary Sentinels of Karnak</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>1338</td>
      <td>262</td>
      <td>5.106870</td>
    </tr>
    <tr>
      <th>11838</th>
      <td>Bal-Sagoth</td>
      <td>Battle Magic</td>
      <td>Blood Slakes the Sand at the Circus Maximus</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>1280</td>
      <td>533</td>
      <td>2.401501</td>
    </tr>
    <tr>
      <th>81809</th>
      <td>Redemption</td>
      <td>Redemption</td>
      <td>Something Wicked This Way Comes</td>
      <td>Progressive Metal</td>
      <td>1182</td>
      <td>1466</td>
      <td>0.806276</td>
    </tr>
    <tr>
      <th>82509</th>
      <td>Rhapsody of Fire</td>
      <td>From Chaos to Eternity</td>
      <td>Heroes of the Waterfalls' Kingdom</td>
      <td>Symphonic Power Metal</td>
      <td>1165</td>
      <td>1172</td>
      <td>0.994027</td>
    </tr>
    <tr>
      <th>15057</th>
      <td>Blind Guardian</td>
      <td>A Night at the Opera</td>
      <td>And Then There Was Silence</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>1088</td>
      <td>846</td>
      <td>1.286052</td>
    </tr>
  </tbody>
</table>
</div>



Sorting by unique word counts shows a similar picture.

### Songs with highest word density

Again "The Obsidian Crown Unbound" tops the charts for highest number of words per second, however at second place, is ["The Ghosts of Christmas Eve"](https://youtu.be/bT4ruFp5U2w), the two-minute intro track to The Christmas Attic by Trans-Siberian Orchestra. Most of the other tracks on this table are short, typically less than a minute.


```python
df_r_songs.sort_values('song_words_per_second')[:-10:-1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band_name</th>
      <th>album_name</th>
      <th>song_name</th>
      <th>band_genre</th>
      <th>song_word_count</th>
      <th>song_seconds</th>
      <th>song_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11862</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>The Obsidian Crown Unbound</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>2391</td>
      <td>358</td>
      <td>6.678771</td>
    </tr>
    <tr>
      <th>103209</th>
      <td>Trans-Siberian Orchestra</td>
      <td>The Christmas Attic</td>
      <td>The Ghosts of Christmas Eve</td>
      <td>Orchestral/Progressive Rock/Metal</td>
      <td>838</td>
      <td>135</td>
      <td>6.207407</td>
    </tr>
    <tr>
      <th>61709</th>
      <td>Macabre</td>
      <td>Gloom</td>
      <td>I Need to Kill</td>
      <td>Thrash/Death Metal/Grindcore</td>
      <td>209</td>
      <td>36</td>
      <td>5.805556</td>
    </tr>
    <tr>
      <th>11866</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Unfettering the Hoary Sentinels of Karnak</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>1338</td>
      <td>262</td>
      <td>5.106870</td>
    </tr>
    <tr>
      <th>84364</th>
      <td>S.O.D.</td>
      <td>Bigger than the Devil</td>
      <td>Charlie Don't Cheat</td>
      <td>Hardcore/Crossover/Thrash Metal</td>
      <td>113</td>
      <td>25</td>
      <td>4.520000</td>
    </tr>
    <tr>
      <th>80005</th>
      <td>Putrid Pile</td>
      <td>Paraphiliac Perversions</td>
      <td>Toxic Shock Therapy</td>
      <td>Brutal Death Metal</td>
      <td>18</td>
      <td>4</td>
      <td>4.500000</td>
    </tr>
    <tr>
      <th>17421</th>
      <td>Brujeria</td>
      <td>Raza odiada</td>
      <td>Sesos humanos (Sacrificio IV)</td>
      <td>Death Metal/Grindcore</td>
      <td>310</td>
      <td>75</td>
      <td>4.133333</td>
    </tr>
    <tr>
      <th>70012</th>
      <td>Napalm Death</td>
      <td>Scum</td>
      <td>You Suffer</td>
      <td>Hardcore Punk (early), Grindcore/Death Metal (...</td>
      <td>4</td>
      <td>1</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>23586</th>
      <td>Circle of Dead Children</td>
      <td>Human Harvest</td>
      <td>White Trash Headache</td>
      <td>Brutal Death Metal, Grindcore</td>
      <td>24</td>
      <td>6</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Word counts by album

Grouping song lyrics by album shows Blind Guardian's 75-minute [Twilight Orchestra: Legacy of the Dark Lands](https://en.wikipedia.org/wiki/Legacy_of_the_Dark_Lands) coming out on top, even outstripping all of Bal-Sagoth's albums on raw word counts. The list of highest word counts per second mostly consists of Bal-Sagoth and very short albums, with [Waste 'Em All](https://en.wikipedia.org/wiki/Waste_%27Em_All) by Municipal Waste topping the chart. Savatage's [The Wake of Magellan] is the most word-dense album that is anywhere near an hour long.


```python
album_lyrics = df_r.groupby('album_name').song_darklyrics.sum()
album_word_counts = album_lyrics.str.split().apply(len)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
album_word_counts.hist(bins=np.logspace(2, 4, 30), ax=ax)
ax.set_xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel("word count")
ax.set_ylabel("number of albums")
ax.set_title("Words per album")
plt.show()
```


![png](lyrics0_files/lyrics0_21_0.png)



```python
df_r_albums = pd.concat((
    df_r_songs.groupby(['band_name', 'album_name']).first()[['band_genre']],
    df_r_songs.groupby(['band_name', 'album_name'])[['song_word_count', 'song_seconds']].sum()
), axis=1).reset_index()
df_r_albums.columns = ['band_name', 'album_name', 'band_genre', 'album_word_count', 'album_seconds']
df_r_albums['album_words_per_second'] = df_r_albums.album_word_count / df_r_albums.album_seconds
df_r_albums.loc[df_r_albums['album_words_per_second'] == np.inf, 'album_words_per_second'] = 0
```

### Albums with highest word counts


```python
df_r_albums.sort_values('album_word_count')[:-10:-1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band_name</th>
      <th>album_name</th>
      <th>band_genre</th>
      <th>album_word_count</th>
      <th>album_seconds</th>
      <th>album_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>924</th>
      <td>Blind Guardian</td>
      <td>Twilight Orchestra: Legacy of the Dark Lands</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>9190</td>
      <td>8210</td>
      <td>1.119367</td>
    </tr>
    <tr>
      <th>676</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>7481</td>
      <td>3639</td>
      <td>2.055785</td>
    </tr>
    <tr>
      <th>1514</th>
      <td>Dalriada</td>
      <td>Arany-album</td>
      <td>Folk Metal</td>
      <td>7342</td>
      <td>4503</td>
      <td>1.630469</td>
    </tr>
    <tr>
      <th>675</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>6948</td>
      <td>3157</td>
      <td>2.200824</td>
    </tr>
    <tr>
      <th>5413</th>
      <td>Savatage</td>
      <td>The Wake of Magellan</td>
      <td>Heavy/Power Metal, Progressive Metal/Rock</td>
      <td>5474</td>
      <td>3218</td>
      <td>1.701057</td>
    </tr>
    <tr>
      <th>653</th>
      <td>Ayreon</td>
      <td>The Human Equation</td>
      <td>Progressive Metal/Rock</td>
      <td>5193</td>
      <td>5950</td>
      <td>0.872773</td>
    </tr>
    <tr>
      <th>6293</th>
      <td>Therion</td>
      <td>Beloved Antichrist</td>
      <td>Death Metal (early), Symphonic/Operatic Metal ...</td>
      <td>5073</td>
      <td>9110</td>
      <td>0.556861</td>
    </tr>
    <tr>
      <th>3820</th>
      <td>Machine Head</td>
      <td>Catharsis</td>
      <td>Groove/Thrash Metal, Nu-Metal</td>
      <td>5012</td>
      <td>4457</td>
      <td>1.124523</td>
    </tr>
    <tr>
      <th>6437</th>
      <td>Trans-Siberian Orchestra</td>
      <td>The Christmas Attic</td>
      <td>Orchestral/Progressive Rock/Metal</td>
      <td>4948</td>
      <td>4066</td>
      <td>1.216921</td>
    </tr>
  </tbody>
</table>
</div>



### Albums with highest word density


```python
df_r_albums.sort_values('album_words_per_second')[:-10:-1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band_name</th>
      <th>album_name</th>
      <th>band_genre</th>
      <th>album_word_count</th>
      <th>album_seconds</th>
      <th>album_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>675</th>
      <td>Bal-Sagoth</td>
      <td>Starfire Burning upon the Ice-Veiled Throne of...</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>6948</td>
      <td>3157</td>
      <td>2.200824</td>
    </tr>
    <tr>
      <th>4275</th>
      <td>Municipal Waste</td>
      <td>Waste 'Em All</td>
      <td>Thrash Metal/Crossover</td>
      <td>1757</td>
      <td>848</td>
      <td>2.071934</td>
    </tr>
    <tr>
      <th>676</th>
      <td>Bal-Sagoth</td>
      <td>The Chthonic Chronicles</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>7481</td>
      <td>3639</td>
      <td>2.055785</td>
    </tr>
    <tr>
      <th>4704</th>
      <td>Origin</td>
      <td>Informis Infinitas Inhumanitas</td>
      <td>Technical Brutal Death Metal</td>
      <td>3132</td>
      <td>1712</td>
      <td>1.829439</td>
    </tr>
    <tr>
      <th>1017</th>
      <td>Brujeria</td>
      <td>Raza odiada</td>
      <td>Death Metal/Grindcore</td>
      <td>4295</td>
      <td>2418</td>
      <td>1.776261</td>
    </tr>
    <tr>
      <th>449</th>
      <td>Archspire</td>
      <td>Relentless Mutation</td>
      <td>Technical Death Metal</td>
      <td>3241</td>
      <td>1837</td>
      <td>1.764290</td>
    </tr>
    <tr>
      <th>5413</th>
      <td>Savatage</td>
      <td>The Wake of Magellan</td>
      <td>Heavy/Power Metal, Progressive Metal/Rock</td>
      <td>5474</td>
      <td>3218</td>
      <td>1.701057</td>
    </tr>
    <tr>
      <th>4271</th>
      <td>Municipal Waste</td>
      <td>Hazardous Mutation</td>
      <td>Thrash Metal/Crossover</td>
      <td>2391</td>
      <td>1425</td>
      <td>1.677895</td>
    </tr>
    <tr>
      <th>4421</th>
      <td>Nekrogoblikon</td>
      <td>Heavy Meta</td>
      <td>Melodic Death/Folk Metal</td>
      <td>3579</td>
      <td>2151</td>
      <td>1.663877</td>
    </tr>
  </tbody>
</table>
</div>



# Word counts by band

Surprisingly, Bal-Sagoth's inflated lyric counts do not matter much when comparing entire bands, perhaps due to how short their discography is. The bands with the highest word counts typically have massive discographies, and are usually power metal or heavy metal bands. Again, thrash and grindcore bands with short songs comprise most of the highest words-per-second list.


```python
band_lyrics = df_r.groupby('band_name').song_darklyrics.sum()
band_word_counts = band_lyrics.str.split().apply(len)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
band_word_counts.hist(bins=np.logspace(2, 5, 30), ax=ax)
ax.set_xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel("word count")
ax.set_ylabel("number of bands")
ax.set_title("Words per band")
plt.show()
```


![png](lyrics0_files/lyrics0_28_0.png)



```python
df_r_bands = pd.concat((
    df_r_songs.groupby('band_name').first()['band_genre'],
    df_r_songs.groupby('band_name')[['song_word_count', 'song_seconds']].sum()
), axis=1).reset_index()
df_r_bands.columns = ['band_name', 'band_genre', 'band_word_count', 'band_seconds']
df_r_bands['band_words_per_second'] = df_r_bands.band_word_count / df_r_bands.band_seconds
df_r_bands.loc[df_r_bands['band_words_per_second'] == np.inf, 'band_words_per_second'] = 0
```

### Bands with highest word counts


```python
df_r_bands.sort_values('band_word_count')[:-10:-1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band_name</th>
      <th>band_genre</th>
      <th>band_word_count</th>
      <th>band_seconds</th>
      <th>band_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>288</th>
      <td>Cradle of Filth</td>
      <td>Death Metal (early), Symphonic Black Metal (mi...</td>
      <td>44217</td>
      <td>44097</td>
      <td>1.002721</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>Saxon</td>
      <td>NWOBHM, Heavy Metal</td>
      <td>39090</td>
      <td>53755</td>
      <td>0.727188</td>
    </tr>
    <tr>
      <th>661</th>
      <td>Iron Maiden</td>
      <td>Heavy Metal, NWOBHM</td>
      <td>36899</td>
      <td>52673</td>
      <td>0.700530</td>
    </tr>
    <tr>
      <th>187</th>
      <td>Blind Guardian</td>
      <td>Speed Metal (early), Power Metal (later)</td>
      <td>36836</td>
      <td>38090</td>
      <td>0.967078</td>
    </tr>
    <tr>
      <th>991</th>
      <td>Rage</td>
      <td>Heavy/Speed/Power Metal</td>
      <td>36636</td>
      <td>56064</td>
      <td>0.653467</td>
    </tr>
    <tr>
      <th>589</th>
      <td>Helloween</td>
      <td>Power/Speed Metal</td>
      <td>35116</td>
      <td>49300</td>
      <td>0.712292</td>
    </tr>
    <tr>
      <th>933</th>
      <td>Overkill</td>
      <td>Thrash Metal; Thrash/Groove Metal</td>
      <td>34456</td>
      <td>47540</td>
      <td>0.724779</td>
    </tr>
    <tr>
      <th>1160</th>
      <td>Tankard</td>
      <td>Thrash Metal</td>
      <td>34103</td>
      <td>38702</td>
      <td>0.881169</td>
    </tr>
    <tr>
      <th>675</th>
      <td>Judas Priest</td>
      <td>Heavy Metal</td>
      <td>32308</td>
      <td>51177</td>
      <td>0.631299</td>
    </tr>
  </tbody>
</table>
</div>



### Bands with highest word density


```python
df_r_bands.sort_values('band_words_per_second')[:-10:-1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band_name</th>
      <th>band_genre</th>
      <th>band_word_count</th>
      <th>band_seconds</th>
      <th>band_words_per_second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>193</th>
      <td>Blood Freak</td>
      <td>Death Metal/Grindcore</td>
      <td>6731</td>
      <td>4447</td>
      <td>1.513605</td>
    </tr>
    <tr>
      <th>844</th>
      <td>Municipal Waste</td>
      <td>Thrash Metal/Crossover</td>
      <td>11199</td>
      <td>7479</td>
      <td>1.497393</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Archspire</td>
      <td>Technical Death Metal</td>
      <td>7676</td>
      <td>5189</td>
      <td>1.479283</td>
    </tr>
    <tr>
      <th>211</th>
      <td>Brujeria</td>
      <td>Death Metal/Grindcore</td>
      <td>10288</td>
      <td>7045</td>
      <td>1.460326</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Acrania</td>
      <td>Brutal Deathcore</td>
      <td>2430</td>
      <td>1674</td>
      <td>1.451613</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Bal-Sagoth</td>
      <td>Symphonic/Epic Black Metal</td>
      <td>23026</td>
      <td>16021</td>
      <td>1.437239</td>
    </tr>
    <tr>
      <th>1175</th>
      <td>The Berzerker</td>
      <td>Industrial Death Metal/Grindcore</td>
      <td>11133</td>
      <td>8290</td>
      <td>1.342943</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Animosity</td>
      <td>Death Metal/Metalcore/Grindcore</td>
      <td>6090</td>
      <td>4681</td>
      <td>1.301004</td>
    </tr>
    <tr>
      <th>497</th>
      <td>Finntroll</td>
      <td>Blackened Folk Metal</td>
      <td>13981</td>
      <td>11291</td>
      <td>1.238243</td>
    </tr>
  </tbody>
</table>
</div>




```python
min_reviews = 20

bands_popular = sorted(set(df_r[df_r['album_review_num'] > min_reviews].band_name))
df_r_bands_popular = df_r_bands[df_r_bands.band_name.isin(bands_popular)].set_index('band_name', drop=True)

plt.figure(figsize=(14, 8))
xlist, ylist = [], []
for band, row in df_r_bands_popular.iterrows():
    x = row['band_seconds'] / 3600.0
    y = row['band_word_count'] / 1000.0
    xlist.append(x)
    ylist.append(y)
    plt.plot(x, y, 'r.')

res = linregress(df_r_bands.band_seconds / 3600.0, df_r_bands.band_word_count / 1000.0)
xline = np.linspace(0, df_r_bands_popular.band_seconds.max() / 3600.0)
yline = xline * res.slope + res.intercept
plt.plot(xline, yline, label='full dataset linear fit')

texts = []
for x, y, band in zip(xlist, ylist, df_r_bands_popular.index):
    texts.append(plt.text(x, y, band, fontsize=12))
adjust_text(texts)

plt.xlabel('Discography length (hours)')
plt.ylabel('Total word count (thousands)')
plt.legend(fontsize=14)
plt.show()
```


![png](lyrics0_files/lyrics0_34_0.png)


# Word counts by genre

Although there are some noticeable trends in the word counts of genres, overall the distributions of word counts and song lengths per genre are quite broad, perhaps overlapping too much to be of any use for predictions later on.


```python
df_genre_songs = df_r[['band_name', 'album_name', 'song_name'] + [f"genre_{genre}" for genre in top_genres_1pct]].copy()
df_genre_songs['song_word_count'] = song_word_counts
df_genre_songs['song_seconds'] = song_seconds

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
violindata = []
for genre in top_genres_1pct:
    df_genre = df_genre_songs[df_genre_songs['genre_' + genre] == 1]
    violindata.append((genre, df_genre['song_word_count']))
violindata.sort(key=lambda x: -np.median(x[1]))
sns.violinplot(data=[x[1] for x in violindata], cut=0, orient='h', color='c')
ax.set_yticklabels([x[0] for x in violindata])
ax.set_xlim(0, 500)
ax.set_title("Words per song")
ax.set_xlabel("word count")
plt.show()
```


![png](lyrics0_files/lyrics0_36_0.png)



```python
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
violindata = []
for genre in top_genres_1pct:
    df_genre = df_genre_songs[df_genre_songs['genre_' + genre] == 1].copy()
    df_genre['song_words_per_second'] = df_genre['song_word_count'] / df_genre['song_seconds']
    df_genre.loc[df_genre['song_words_per_second'] == np.inf, 'song_words_per_second'] = 0
    violindata.append((genre, df_genre['song_words_per_second']))
violindata.sort(key=lambda x: -np.median(x[1]))
sns.violinplot(data=[x[1] for x in violindata], cut=0, orient='h', color='c')
ax.set_yticklabels([x[0] for x in violindata])
ax.set_title("Words per second")
ax.set_xlabel("word count")
plt.show()
```


![png](lyrics0_files/lyrics0_37_0.png)



```python
plt.figure(figsize=(14, 8))
xlist, ylist = [], []
for genre in top_genres_1pct:
    df_genre = df_genre_songs[df_genre_songs['genre_' + genre] == 1].copy()
    x = df_genre['song_seconds'].mean() / 60.0
    y = df_genre['song_word_count'].mean()
    xlist.append(x)
    ylist.append(y)
    plt.plot(x, y, 'r.', ms=10, label=genre)
texts = []
for x, y, genre in zip(xlist, ylist, top_genres_1pct):
    texts.append(plt.text(x, y , genre, fontsize=12))
adjust_text(texts)
plt.xlabel('Average song length (minutes)')
plt.ylabel('Average words per song')
plt.show()
```


![png](lyrics0_files/lyrics0_38_0.png)


### 95% contours for top-five genres


```python
df_genre_10pct_songs = df_rr[['band_name', 'album_name', 'song_name'] + [f"genre_{genre}" for genre in top_genres_10pct]].copy()
df_genre_10pct_songs['song_word_count'] = df_rr.song_darklyrics.str.split().apply(len)
df_genre_10pct_songs['song_seconds'] = to_seconds(df_rr)
```


```python
plt.figure(figsize=(14, 8))
ax = plt.gca()
colors = sns.color_palette()
handles = []
for i, genre in enumerate(top_genres_10pct):
    df_genre = df_genre_10pct_songs[df_genre_10pct_songs['genre_' + genre] == 1].copy()
    x = np.log10(df_genre['song_seconds'] / 60.0)
    y = np.log10(df_genre['song_word_count'])
    pos = (x > 0) & (y > 0)
    sns.kdeplot(x=x[pos], y=y[pos], levels=[0.05, 1], ax=ax)
    handles.append(Patch(facecolor=colors[i % len(colors)], label=genre))
ax.set_xticklabels([f"{10**tick:.0f}" for tick in ax.get_xticks()])
ax.set_yticklabels([f"{10**tick:.0f}" for tick in ax.get_yticks()])
ax.set_xlabel('Song length (minutes)')
ax.set_ylabel('Word count per song')
ax.legend(handles=handles)
plt.show()
```

    C:\Users\philippe\Anaconda3\envs\metallyrics\lib\site-packages\pandas\core\series.py:726: RuntimeWarning: divide by zero encountered in log10
      result = getattr(ufunc, method)(*inputs, **kwargs)
    C:\Users\philippe\Anaconda3\envs\metallyrics\lib\site-packages\pandas\core\series.py:726: RuntimeWarning: divide by zero encountered in log10
      result = getattr(ufunc, method)(*inputs, **kwargs)
    C:\Users\philippe\Anaconda3\envs\metallyrics\lib\site-packages\ipykernel_launcher.py:12: UserWarning: FixedFormatter should only be used together with FixedLocator
      if sys.path[0] == '':
    C:\Users\philippe\Anaconda3\envs\metallyrics\lib\site-packages\ipykernel_launcher.py:13: UserWarning: FixedFormatter should only be used together with FixedLocator
      del sys.path[0]
    


![png](lyrics0_files/lyrics0_41_1.png)

