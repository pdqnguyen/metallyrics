# Lyric Analysis Part 0 - Data summary

Before analyzing the data, some preprocessing steps need to be taken to condition the data. These are handled by the 
`analyses/lyrics/scripts/preprocessing.py` script.

### Filtering out songs

Songs that are mostly English and songs whose lyrics were removed due to copyright (thus leaving a "copyright claimed" 
message) are filtered out.

### Reduced dataset

For lyrical analyses the dataset is reduced to just a column of lyrics (which will become the feature vector upon some 
transformation to a quantitative representation) for each song and columns for the most popular genres (the 
target/label vectors). These are the genres that appear at least once in isolation, i.e. not accompanied by any other 
genre, and that appear in some minimum percentage of songs. For example, the "black" metal label can appear on bands 
with or without other genres, but a label like "atmospheric" never appears on its own despite being fairly popular, 
usually because it is more of an adjective to denote subgenres like atmospheric black metal; thus "black" is included 
in the reduced label space but "atmospheric" is not. This reduces the genres to a more manageable set: five genres if 
the minimum occurrence requirement is set to 10%, and thirteen if set to 1%.

A five-genre set would be easier to handle but leaves quite a few holes in the label space, because doom metal, 
metalcore, folk metal, and many other fairly popular genres are being omitted that may not be covered by any of the 
five labels. The larger label set covers just about all the most important genres, but because eight of them occur in 
fewer than 10% of all songs, they will force greater class imbalance which will adversely affect attempts at applying 
binary classification models later on. For the sake of comparison, both reduced datasets are saved here, but the rest 
of this exploratory analysis only looks at the 1% dataset, while the 10% dataset is reserved for modeling. Each dataset 
is saved in its raw form and in a truncated (ML-ready) form containing only the lyrics and genre columns.

### General statistics

![Histogram of word count per song](../output/basic_plots/song_words.png)

![Histogram of word count per album](../output/basic_plots/album_words.png)

![Histogram of word count per band](../output/basic_plots/band_words.png)

![Word count per song split by genre](../output/basic_plots/genre_words.png)

![Word rate per song split by genre](../output/basic_plots/genre_word_rate.png)