# Heavy metal lyrics and reviews

## Overview

This repository contains analyses of heavy metal artists and their lyrical content.
The core data set combines artist information, including genre labels, and album reviews from
[The Metal-Archives](https://www.metal-archives.com) (MA) and song lyrics from [DarkLyrics](http://www.darklyrics.com)
(DL). The data collection begins with the `metallum_ids.py` script, which reads through the complete list of
[album reviews sorted by artist](https://www.metal-archives.com/review/browse/by/alpha) in order to build a csv table 
of artist names and id numbers for artists with at least one album review (`/data/ids.csv`). Artist information and
full-text album reviews are then scraped by `metallum.py` and saved into json files (`/data/bands.zip`). The DL 
scraping tool `darklyrics_fast.py` searches DL for the corresponding album lyrics and adds them to the json files. 
Finally, the data set is split by `create_dataframes.py` into a csv table of album reviews and a csv table of song 
lyrics (`/data/data.zip`).

## Notebooks & analyses

The analyses below provide insights on the history of heavy metal albums, and linguistic properties of metal lyrics.


> [Exploration of artists and album reviews](https://github.com/pdqnguyen/metallyrics/blob/master/analyses/reviews/reviews1.ipynb)
>
> A data-driven discussion of the history and global demographics of the heavy metal music industry and its many
> genres. This notebook also provides statistical insights on the sentiments of MA users as expressed through online
> album reviews.

> [Lyrics data exploration](https://github.com/pdqnguyen/metallyrics/blob/master/analyses/lyrics/markdown/lyrics0.md)
> 
> Brief overview of the lyrics data set.

> [Lexical diversity measures](https://github.com/pdqnguyen/metallyrics/blob/master/analyses/lyrics/markdown/lyrics1.md)
> 
> Comparison of lexical diversity measures and what they tell us about artists and genres.

> [Word clouds](https://github.com/pdqnguyen/metallyrics/blob/master/analyses/lyrics/markdown/lyrics2.md)
> 
> Concise visualizations of song lyrics from different genres.

> [Machine learning notebook](https://github.com/pdqnguyen/metallyrics/blob/master/analyses/lyrics/notebooks/song-lyrics-multi-genre-bow.ipynb)
> 
> This notebook presents the multi-label problem of genre classification based on lyrics. Different approaches
> and preprocessing steps are discussed, and various machine learning models are compared via cross-validation
> to demonstrate possible solutions.

## Machine learning scripts

For the genre classifier tool (see link at the bottom of page), a number of machine learning models were tuned and
trained to assign genre tags to text inputs of arbitrary length. As discussed in the machine learning notebook above,
these models are incorporated into pipelines that also vectorize (and oversample, when training) the data. The
relevant scripts are located in `analyses/lyrics/scripts` and are configured by the corresponding `.yaml` files in
`analyses/lyrics`. The `genre_classification_tuning.py` script tunes the models using cross-validation to determine
optimal hyperparameters. The `genre_classification_train.py` script is used to train the model, given those optimal
hyperparameters, and `genre_classification_test.py` can be used to test the pipeline for functionality before
deploying it to the genre classifier tool.

## Interactive webpages

Source code for these webpages can be found in the [pdqnguyen/metallyrics-web](https://github.com/pdqnguyen/metallyrics-web) repository. 

> [Interactive data dashboard](https://metal-lyrics-feature-plots.herokuapp.com/)
> 
> Explore the lyrics and album reviews data sets through interactive scatter plots and swarm plots.

> [Network graph of heavy metal bands](https://metal-lyrics-network-graph.herokuapp.com/)
> 
> See how genre associations and lyrical similarity connect the disparate world of heavy metal artists.

> [Interactive genre classifier tool](https://metal-lyrics-genre-classifier.herokuapp.com/)
> 
> Enter any text you want and see what heavy metal genres it fits in best.
