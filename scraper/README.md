# Scraping Metal-Archives and DarkLyrics

Instead of scraping the full dataset off of DL, I only collect lyrics for bands who exist on MA, since MA is where I get the genre tags needed for the genre classification task. However, I do collect as much from MA as possible for the reviews data set. So the data set scraping grabs bands/reviews from MA first, then populates the resulting .json files with lyrics from DL when available. The workflow of scripts is as follows:

1. `metallum_ids.py` goes through the band lists on MA to get the URL band ID of each band.
2. `metallum.py` uses those URLS to fetch metadata (genres, year formed, etc.) and reviews (text and rating) for each band. Results are saved to .json files.
3. `darklyrics_urls.py` searches the DL catalog for bands whose names match the ones in the MA .json files. It saves the DL urls for these bands to a .csv file.
4. `darklyrics.py` fetches lyrics from DL for each band, making sure to match albums to the ones fetched by `metallum.py`. The lyrics are added to the existing MA .json files.
