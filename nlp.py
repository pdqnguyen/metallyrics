import glob
import os
import re


STOPWORDS_DIR = os.path.abspath('stopwords')


def get_stopwords(stopword_dir=STOPWORDS_DIR):
    stopwords = set()
    filenames = glob.glob(os.path.join(stopword_dir, '*'))
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for word in lines:
            if re.match('^\w+$', word):
                stopwords.add(word)
    return stopwords
