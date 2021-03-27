# Heavy metal lyrics and reviews

Table of Contents

- [Section 1: Background](#Section-1:-Background)

- [Section 2: Collecting the data](#Section-2:-Data)

- [Section 3: Reviews](#Section-3:-Reviews)

    - [Section 3a: Exploring the data](#Section-3a:-Exploring-the-data)

    - [Section 3b: Recommending albums based on reviews](#Section-3b:-Recommending-albums-based-on-reviews)

- [Section 4a: Exploring the data](#Section-4a:-Exploring-the-data)

    - [Section 4b: Labeling genres based on lyrics](#Section-4b:-Labeling-genres-based-on-lyrics)

    - [Section 4c: Generating lyrics by genre](#Section-4c:-Generating-lyrics-by-genre)


## Section 1: Background

## Section 2: Data

The core data set combines artist information, including genre labels, and album reviews from
[The Metal-Archives](https://www.metal-archives.com) (MA) and song lyrics from [DarkLyrics](http://www.darklyrics.com)
(DL). The data collection begins with the `metallum_ids.py` script, which reads through the complete list of
[album reviews sorted by artist](https://www.metal-archives.com/review/browse/by/alpha) in order to build a csv table 
of artist names and id numbers for artists with at least one album review (`/data/ids.csv`). Artist information and
full-text album reviews are then scraped by `metallum.py` and saved into json files (`/data/bands.zip`). The DL 
scraping tool `darklyrics_fast.py` searches DL for the corresponding album lyrics and adds them to the json files. 
Finally, the data set is split by `create_dataframes.py` into a csv table of album reviews and a csv table of song 
lyrics (`/data/data.zip`).

## Section 3: Reviews

### Section 3a: Exploring the data

### Section 3b: Recommending albums based on reviews

## Section 4: Lyrics

### Section 4a: Exploring the data

### Section 4b: Labeling genres based on lyrics

### Section 4c: Generating lyrics by genre