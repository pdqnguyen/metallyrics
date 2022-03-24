import sys
import yaml
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix
from scipy.sparse.lil import lil_matrix

from keras.models import Sequential
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.problem_transform import BinaryRelevance

from nltk.corpus import stopwords

from nlp import tokenize


plt.style.use('seaborn')
sns.set(font_scale=2)


def get_config():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('-n', '--names', nargs='+', help='names of specific models in config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    for key in ['input', 'output', 'vectorizer', 'resampler', 'models']:
        if key not in cfg.keys():
            raise KeyError(f"missing field {key} in {args.config}")
    if args.names is not None:
        cfg['models'] = [model for model in cfg['models'] if model['name'] in args.names]
    return cfg


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def tokenizer(s):
    tokens = tokenize(s.strip(), english_only=True)
    tokens = [t for t in tokens if len(t) >= 4]
    return tokens


def get_class(name, module):
    exec(f"from {module} import {name}")
    return eval(name)


def init_model(cfg, *args, **default_kwargs):
    try:
        cls = eval(cfg['name'])
    except NameError:
        exec(f"from {cfg['module']} import {cfg['name']}")
        cls = eval(cfg['name'])
    kwargs = default_kwargs.copy()
    kwargs.update(cfg['params'])
    model = cls(*args, **kwargs)
    return model


def build_pipeline(vectorizer_cfg, resampler_cfg, est_cfg):
    vectorizer_default_kwargs = dict(
        stop_words=stopwords.words('english'),
        tokenizer=tokenizer
    )
    vectorizer = init_model(vectorizer_cfg, **vectorizer_default_kwargs)
    resampler = init_model(resampler_cfg)
    args = ()
    if est_cfg['name'] == 'KerasClassifier':
        args += (create_keras_model,)
        # est_cfg['input_dim'] = vectorizer_cfg['params']['max_features']
        pad_features = True
        classifier = init_model(est_cfg, *args)
    else:
        pad_features = False
        est = init_model(est_cfg, *args)
        classifier = BinaryRelevance(est, require_dense=[False, True])
    pipeline = NLPipeline(
        vectorizer=vectorizer,
        resampler=resampler,
        classifier=classifier,
        pad_features=pad_features,
    )
    return pipeline


def init_tf_session(seed=0):
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    return


def create_keras_model(input_dim=None, output_dim=None, architecture=None):
    assert isinstance(architecture, list), "`architecture` must be list of integers"
    model = Sequential()
    model.add(layers.Dense(architecture[0], input_dim=input_dim, activation='relu'))
    if len(architecture) > 1:
        for x in architecture[1:]:
            model.add(layers.Dropout(rate=0.2))
            model.add(layers.Dense(x, activation='relu'))
    model.add(layers.Dense(output_dim, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    print(model.summary())
    return model


class NLPipeline:
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
    def __init__(self, vectorizer, resampler, classifier, pad_features=False):
        self.vectorizer = vectorizer
        self.resampler = resampler
        self.classifier = classifier
        self.pad_features = pad_features
        self.padding = 0
        self.threshold = None
        self.labels = None

    @property
    def features(self):
        feature_names = self.vectorizer.get_feature_names()
        if self.pad_features:
            feature_names + [''] * self.padding
        return

    def apply_padding(self, X):
        if self.padding > 0:
            padding_array = np.zeros((X.shape[0], self.padding))
            X = np.concatenate((X, padding_array), axis=1)
        return X

    def fit(self, X, y, labels=None, verbose=0):
        self.labels = labels
        X_v = self.vectorizer.fit_transform(X).toarray()
        if self.pad_features:
            self.padding = self.vectorizer.max_features - len(self.vectorizer.get_feature_names())
            X_v = self.apply_padding(X_v)
        X_r, y_r = self.resampler.fit_resample(X_v, y)
        if isinstance(self.classifier, KerasClassifier):
            callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=2,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=True,
            )
            self.classifier.fit(X_r, y_r, validation_split=0.2, callbacks=callback, verbose=verbose)
        else:
            self.classifier.fit(X_r, y_r)
        return self

    def predict(self, X):
        X_v = self.vectorizer.transform(X).toarray()
        X_v = self.apply_padding(X_v)
        if self.padding > 0:
            padding_array = np.zeros((X_v.shape[0], self.padding))
            X_v = np.concatenate((X_v, padding_array), axis=1)
        try:
            y_p = self.classifier.predict_proba(X_v)
        except AttributeError:
            y_p = self.classifier.predict(X_v)
        if (
                isinstance(y_p, csr_matrix) or
                isinstance(y_p, lil_matrix)
        ):
            y_p = y_p.toarray()
        return y_p

    def set_threshold(self, threshold):
        self.threshold = threshold
        return

    def classify_text(self, text, verbose=False):
        X_test = np.array([' '.join(text.lower().split())])
        prob = self.predict(X_test)[0]
        if self.threshold is not None:
            pred = prob > self.threshold
        else:
            pred = prob > 0.5
        if self.labels is not None:
            labels = self.labels
        else:
            labels = range(len(pred))
        results = [(label, prob[i], pred[i]) for i, label in enumerate(labels)]
        results.sort(key=lambda x: 1 - x[1])
        if verbose:
            print("Classification:")
            if results[0][2] < 1:
                print("NONE")
            else:
                print(", ".join([res[0].upper() for res in results if res[2] > 0]))
            print("\nIndividual label probabilities:")
            for res in results:
                print("{:<10s}{:>5.2g}%".format(res[0], 100 * res[1]))
        return results


def multilabel_pipeline_cross_val(pipeline, X, y, labels=None, n_splits=3, verbose=0):
    """Multi-label pipeline cross-validation

    Parameters
    ----------
    pipeline : `sklearn.pipeline.Pipeline` or custom pipeline
        Must have .fit and .predict methods

    X : array-like

    y : array-like
        (n_samples x n_labels)

    labels : array-like
        Label names (numerical if Default = None)

    n_splits : int
        Number of cross-validation splits (Default = 3)

    Returns
    -------
    mlc : `multilabel.MultiLabelClassification`
        Multi-label classification results

    folds : list
        (train_idx, valid_idx) pair for each CV fold
    """
    kfold = IterativeStratification(n_splits=n_splits, order=1, random_state=None)
    pred = np.zeros_like(y, dtype=float)
    thresh_folds = np.zeros((y.shape[1], n_splits))
    for i, (train_idx, valid_idx) in enumerate(kfold.split(X, y)):
        if verbose > 0:
            print(f"\n--------\nFold {i+1}/{kfold.n_splits}")
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        pipeline.fit(X_train, y_train, labels=labels, verbose=verbose)
        valid_pred = pipeline.predict(X_valid)
        pred[valid_idx] = valid_pred
        mlc_valid = MultiLabelClassification(y_valid, valid_pred, labels=labels)
        thresh_folds[:, i] = mlc_valid.best_thresholds('gmean')
        if verbose > 0:
            mlc_valid.print_report(full=(verbose > 1))
    threshold = thresh_folds.mean(axis=1)
    mlc = MultiLabelClassification(
        y, pred=pred, labels=labels, threshold=threshold)
    if verbose > 0:
        print("\n------------------------\nCross-validation results")
        mlc.print_report(full=True)#(verbose > 1))
    return mlc


class MultiLabelClassification:
    """Multi-label classification results and evaluation metrics.

    Parameters
    ----------
    true : `numpy.ndarray`
        True values (n_samples, n_labels).

    pred : `numpy.ndarray`
        Predicted probabilities (n_samples, n_labels).

    pred_class : `numpy.ndarray`
        Classification results (n_samples, n_labels).

    labels : array-like
        Label names (str).

    threshold : float or array-like
        If float, `thresh` is a decision threshold for all labels.
        If array-like, `thresh` must be length n_labels, with each
        value a decision threshold for that respective label.


    Attributes
    ----------
    n_samples : int
        Number of samples (rows in `self.true`).

    n_labels : int
        Number of labels (columns in `self.true`).

    accuracy_score : float
        Number of labels in common / overall labels (true and predicted).

    precision_score : float
        Proportion of predicted labels that are correct.

    recall_score : float
        Proportion of true labels that were predicted.

    f1_score : float
        Harmonic mean of precision_score and recall_score.

    hamming_loss : float
        Symmetric difference b/w pred and true labels (true XOR pred).


    Methods
    -------
    print_report
    best_thresholds
    roc_auc_score
    plot_roc_curve
    plot_precision_recall_curve
    to_csv
    from_csv
    """

    def __init__(
            self,
            true,
            pred=None,
            pred_class=None,
            labels=None,
            threshold=0.5
    ):
        self.true = true.astype(int)
        self.pred = pred
        self.threshold = threshold
        if pred_class is None:
            pred_class = np.zeros_like(self.pred, dtype=int)
            if hasattr(self.threshold, '__iter__'):
                thresh_tile = np.ones_like(self.true) * self.threshold
            else:
                thresh_tile = np.tile(self.threshold, (self.true.shape[0], 1))
            pred_class[self.pred > thresh_tile] = 1
        self.pred_class = pred_class
        self.n_samples, self.n_labels = self.true.shape
        if labels is not None:
            if len(labels) == self.n_labels:
                self.labels = np.array(labels, dtype='object')
            else:
                raise ValueError(
                    f"len(labels)={len(labels)} does not match "
                    f"true.shape[1]={self.n_labels}")
        else:
            self.labels = np.arange(self.true.shape[1]).astype(str).astype('object')

    @property
    def __intersection(self):
        return self.true * self.pred_class

    @property
    def __union(self):
        return np.minimum(1, self.true + self.pred_class)

    @property
    def accuracy_score(self):
        return np.nanmean(self.__intersection.sum(1) / self.__union.sum(1))

    @property
    def precision_score(self):
        return np.nanmean(self.__intersection.sum(1) / self.pred_class.sum(1))

    @property
    def recall_score(self):
        return np.nanmean(self.__intersection.sum(1) / self.true.sum(1))

    @property
    def f1_score(self):
        prec = self.precision_score
        rec = self.recall_score
        return 2 * prec * rec / (prec + rec)

    @property
    def hamming_loss(self):
        delta = np.zeros(self.true.shape[0])
        for i in range(delta.shape[0]):
            delta[i] = np.sum(self.true[i] ^ self.pred_class[i])
        return delta.mean()

    @property
    def roc_auc_score(self):
        """Area under receiver operating characteristic (ROC) curve.
        """
        auc = np.zeros(len(self.labels))
        for i, label in enumerate(self.labels):
            auc[i] = roc_auc_score(self.true[:, i], self.pred[:, i])
        return auc

    def print_report(self, full=False):
        """Print results of classification.
        """
        np.seterr(divide='ignore', invalid='ignore')
        if full:
            print("\nBinary classification metrics:")
        metrics = [
            'balanced_accuracy_score', 'precision_score',
            'recall_score', 'f1_score']
        exec(f"from sklearn.metrics import {', '.join(metrics)}")
        scores = {metric: np.zeros(self.n_labels) for metric in metrics}
        for i, label in enumerate(self.labels):
            if full:
                print(f"\nlabel: {label}")
            true_i = self.true[:, i]
            pred_i = self.pred_class[:, i]
            for metric in metrics:
                score = eval(f"{metric}(true_i, pred_i)")
                scores[metric][i] = score
                if full:
                    print(f"  {metric.replace('_score', '')[:19]:<20s}"
                          f"{score:.3f}")
            cfm = confusion_matrix(true_i, pred_i)
            if full:
                print("  confusion matrix:")
                print(f"  [[{cfm[0, 0]:6.0f} {cfm[0, 1]:6.0f}]\n"
                      f"   [{cfm[1, 0]:6.0f} {cfm[1, 1]:6.0f}]]")
        print(f"\nAverage binary classification scores:")
        for metric in metrics:
            avg = scores[metric].mean()
            std = scores[metric].std()
            print(f"  {metric.replace('_score', '')[:19]:<20s}"
                  f"{avg:.2f} +/- {std * 2:.2f}")
        print("\nMulti-label classification metrics:")
        print(f"  accuracy      {self.accuracy_score:.2f}")
        print(f"  precision     {self.precision_score:.2f}")
        print(f"  recall        {self.recall_score:.2f}")
        print(f"  f1            {self.f1_score:.2f}")
        print(f"  hamming loss  {self.hamming_loss:.2f}")
        auc_scores = self.roc_auc_score
        print(f"\nROC AUC scores:")
        for label, auc_score in zip(self.labels, auc_scores):
            print(f"  {label:<10s}: {auc_score:.3f}")
        print(f"  macro-avg : {np.mean(auc_scores):.3f} "
              f"+/- {np.std(auc_scores):.3f}")
        return

    def best_thresholds(self, metric='gmean', fbeta=1):
        """Determine best thresholds by maximizing geometric mean
        or f_beta score.
        """
        best = np.zeros(len(self.labels))
        for i, label in enumerate(self.labels):
            true, pred = self.true[:, i], self.pred[:, i]
            if metric == 'gmean':
                fpr, tpr, thresholds = roc_curve(true, pred)
                gmean = np.sqrt(tpr * (1 - fpr))
                best[i] = thresholds[gmean.argmax()]
            elif metric == 'fscore':
                prec, rec, thresholds = precision_recall_curve(true, pred)
                fscore = ((1 + fbeta**2) * prec * rec) / ((fbeta**2 * prec) + rec)
                best[i] = thresholds[fscore.argmax()]
        return best

    def plot_roc_curve(self):
        """Plot receiver-operating characteristic (ROC) curve.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for i, label in enumerate(self.labels):
            true = self.true[:, i]
            pred = self.pred[:, i]
            fpr, tpr, thresholds = roc_curve(true, pred)
            ax.step(fpr, tpr, label=label)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_aspect('equal')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title("ROC curve")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig

    def plot_precision_recall_curve(self):
        """Plot precision and recall against decision threshold.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for i, label in enumerate(self.labels):
            true, pred = self.true[:, i], self.pred[:, i]
            prec, rec, thresholds = precision_recall_curve(true, pred)
            line = ax.plot(thresholds, prec[:-1], label=label)
            ax.plot(thresholds, rec[:-1], ":", color=line[0].get_color())
        ax.set_aspect('equal')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title("Precision and recall scores")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Score")
        ax.text(0.01, 0.01, "solid lines show precision score\n"
                "dotted lines show recall score", size=16)
        ax.legend(loc='upper right')
        ax.grid(True)
        fig.tight_layout()
        return fig

    def to_csv(self, filename):
        """Save true labels, probabilities, and predictions to CSV.
        """
        data = {}
        for i, label in enumerate(self.labels):
            data[f"{label}_true"] = self.true[:, i]
            data[f"{label}_pred"] = self.pred[:, i]
            data[f"{label}_pred_class"] = self.pred_class[:, i]
        df = pd.DataFrame.from_dict(data)
        df.to_csv(filename, index=False)
        return

    @classmethod
    def from_csv(cls, filename):
        """Load classification from CSV.
        """
        data = pd.read_csv(filename)
        cols = data.columns
        true = data[[c for c in cols if c[-4:] == 'true']].values
        pred = data[[c for c in cols if c[-4:] == 'pred']].values
        labels = [c.replace('_true', '') for c in cols if c[-4:] == 'true']
        pred_class = data[[c for c in cols if c[-10:] == 'pred_class']].values
        new = cls(true, pred, pred_class=pred_class, labels=labels)
        return new


def plot_feature_importances(clf, vocab):
    if hasattr(clf, 'feature_importances_'):
        fi = clf.feature_importances_
    elif hasattr(clf, 'feature_log_prob_'):
        fi = clf.feature_log_prob_[1, :]
    elif hasattr(clf, 'coef_'):
        fi = clf.coef_[0]
    else:
        raise AttributeError(f"Object {clf.__name__} has no feature importance attribute")
    fi_top = fi.argsort()[-10:]
    x_vals = range(len(fi_top))
    fig = plt.figure(figsize=(8, 5))
    plt.bar(x_vals, fi[fi_top])
    plt.xticks(x_vals, vocab[fi_top], rotation=45)
    return fig
