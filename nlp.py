import glob
import os
import re

from nltk.corpus import words as nltk_words


STOPWORDS_DIR = os.path.abspath('stopwords')
ENGLISH_WORDS = set(nltk_words.words())


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


def tokenize(s, lower=True, alpha_only=True, english_only=True, stopwords=None):
    words = s.split()
    if lower:
        words = [word.lower() for word in words]
    if alpha_only:
        word = [word for word in words if word.isalpha()]
    if english_only:
        words = [word for word in words if word in ENGLISH_WORDS]
    if stopwords:
        words = [word for word in words if word.lower() not in stopwords]
    return words
