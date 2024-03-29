import scipy

from nltk.corpus import words as nltk_words
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer


ENGLISH_WORDS = set(nltk_words.words())


def tokenize(s, english_only=False, stopwords=None):
    tokenizer = RegexpTokenizer(r'[\w+\-\']+')
    words = [w.lower()
             .replace("'s", '')
             .replace("in'", 'ing')
             for w in tokenizer.tokenize(s.strip())]
    if english_only:
        new_words = []
        for w in words:
            lemma = wordnet.morphy(w)
            if lemma is not None:
                if lemma in ENGLISH_WORDS:
                    new_words.append(w)
            elif w in ENGLISH_WORDS:
                new_words.append(w)
        words = new_words
    if stopwords is not None:
        words = [w for w in words if w not in stopwords]
    return words


def lemmatize(tokens):
    lemmas = []
    for t in tokens:
        lemma = wordnet.morphy(t)
        if lemma is not None:
            lemmas.append(lemma)
        else:
            lemmas.append(t)
    return lemmas


class Pipeline:
    """Pipeline for NLP classification with vectorization and resampling

    Parameters
    ----------
    vectorizer : transformer object
        Object with `fit_transform` and `transform` methods for vectorizing
        corpus data.

    resampler : resampler object
        Object with fit_resample method for resampling training data in
        `Pipeline.fit`. This can be any under/oversampler from `imblearn`
        for binary or multiclass classification, or `MLSOL` from
        https://github.com/diliadis/mlsol/blob/master/MLSOL.py for multi-label
        classification.

    classifier : estimator object
        Binary, multi-class, or multi-label classifier with a `predict`
        or `predict_proba` method.

    Methods
    -------
    fit(X, y)
        Fit vectorizer and resampler, then train classifier on transformed data.

    predict(X)
        Return classification probabilities (if `self.classifier` has a
        `predict_proba` method, otherwise return predictions using `predict`).
    """
    def __init__(self, vectorizer, resampler, classifier):
        self.vectorizer = vectorizer
        self.resampler = resampler
        self.classifier = classifier

    def fit(self, X, y):
        X_v = self.vectorizer.fit_transform(X)
        X_r, y_r = self.resampler.fit_resample(X_v.toarray(), y)
        self.classifier.fit(X_r, y_r)
        return self

    def predict(self, X):
        X_v = self.vectorizer.transform(X)
        try:
            y_p = self.classifier.predict_proba(X_v)
        except AttributeError:
            y_p = self.classifier.predict(X_v)
        if (
                isinstance(y_p, scipy.sparse.csr.csr_matrix) or
                isinstance(y_p, scipy.sparse.lil.lil_matrix)
        ):
            y_p = y_p.toarray()
        return y_p
