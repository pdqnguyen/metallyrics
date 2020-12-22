from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, multilabel_confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve
)


# Set font scale for all plots
sns.set(font_scale=2)


class BinaryRelevance:
    def __init__(self, classifier, labels, thresholds=None):
        self.classifier = classifier
        self.labels = labels
        self.classifiers_ = []
        self.thresholds = thresholds

    def fit(self, X, y):
        for i, label in enumerate(self.labels):
            print('training binary classifier for label: {}'.format(label))
            clf = deepcopy(self.classifier)
            clf.fit(X, y[:, i])
            self.classifiers_.append((label, clf))

    def predict(self, X):
        return self._predict(X).astype(int)

    def predict_proba(self, X):
        return self._predict(X, return_prob=True)

    def _predict(self, X, return_prob=False):
        y = np.zeros((X.shape[0], len(self.classifiers_)))
        for i, (label, clf) in enumerate(self.classifiers_):
            if return_prob:
                y[:, i] = clf.predict_proba(X)[:, 1]
            elif self.thresholds is not None:
                y[:, i] = (clf.predict_proba(X)[:, 1] > self.thresholds[i])
            else:
                y[:, i] = clf.predict(X)
        return y

    def cross_validate(self, X, y, n_splits=5, verbose=False):
        kfold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=0)
        y_prob, y_valid = np.zeros(y.shape), np.zeros(y.shape)
        for i, label in enumerate(self.labels):
            if verbose:
                print("Training on label '{}'".format(label))
            scores = np.zeros((n_splits, 4))
            confusion_matrices = np.zeros((n_splits, 2, 2))
            for j, (train_idx, valid_idx) in enumerate(kfold.split(X, y[:, i])):
                if verbose:
                    print("Fold {}/{}".format(j + 1, n_splits))
                X_fold_train, y_fold_train = X[train_idx], y[train_idx, i]
                X_fold_valid, y_fold_valid = X[valid_idx], y[valid_idx, i]
                clf = deepcopy(self.classifier)
                clf.fit(X_fold_train, y_fold_train)
                try:
                    y_fold_prob = clf.predict_proba(X_fold_valid)[:, 1]
                except AttributeError:
                    y_fold_prob = clf.predict(X_fold_valid).reshape(-1)
                y_prob[valid_idx, i] = y_fold_prob
                y_valid[valid_idx, i] = y_fold_valid
                y_fold_pred = y_fold_prob.round()
                scores[j, 0] = balanced_accuracy_score(y_fold_valid, y_fold_pred)
                scores[j, 1] = precision_score(y_fold_valid, y_fold_pred)
                scores[j, 2] = recall_score(y_fold_valid, y_fold_pred)
                scores[j, 3] = f1_score(y_fold_valid, y_fold_pred)
                confusion_matrices[j] = confusion_matrix(y_fold_valid, y_fold_pred)
            if verbose:
                avg_accuracy, avg_precision, avg_recall, avg_f1 = scores.mean(axis=0)
                std_accuracy, std_precision, std_recall, std_f1 = scores.std(axis=0)
                print("CV micro-avg scores:")
                print(f"  Accuracy:    {avg_accuracy:.2f} +/- {std_accuracy:.2f}")
                print(f"  Precision:   {avg_precision:.2f} +/- {std_precision:.2f}")
                print(f"  Recall:      {avg_recall:.2f} +/- {std_recall:.2f}")
                print(f"  F1-score:    {avg_f1:.2f} +/- {std_f1:.2f}")
                print(f"  Average confusion matrix:")
                print(confusion_matrices.mean(axis=0) /
                      confusion_matrices.sum() * n_splits)
        return MultiLabelClassification(y_valid, y_prob, labels=self.labels)


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

    thresh : float or array-like
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
            thresh=0.5
    ):
        self.true = true.astype(int)
        self.pred = pred
        self.thresh = thresh
        if pred_class is None:
            pred_class = np.zeros_like(self.pred, dtype=int)
            pred_class[self.pred > self.thresh] = 1
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

    def print_report(self):
        """Print results of classification.
        """
        np.seterr(divide='ignore', invalid='ignore')
        print("\nBinary classification metrics:")
        metrics = [
            'balanced_accuracy_score', 'precision_score',
            'recall_score', 'f1_score']
        exec(f"from sklearn.metrics import {', '.join(metrics)}")
        scores = {metric: np.zeros(self.n_labels) for metric in metrics}
        for i, label in enumerate(self.labels):
            print(f"\nlabel: {label}")
            true_i = self.true[:, i]
            pred_i = self.pred_class[:, i]
            for metric in metrics:
                score = eval(f"{metric}(true_i, pred_i)")
                scores[metric][i] = score
                print(f"  {metric.replace('_score', '')[:19]:<20s}"
                      f"{score:.3f}")
            print("  confusion matrix:")
            cfm = confusion_matrix(true_i, pred_i)
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
        auc_scores = self.roc_auc_score()
        print(f"ROC AUC scores:")
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

    def roc_auc_score(self):
        """Area under receiver operating characteristic (ROC) curve.
        """
        auc = np.zeros(len(self.labels))
        for i, label in enumerate(self.labels):
            auc[i] = roc_auc_score(self.true[:, i], self.pred[:, i])
        return auc

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


def multilabel_pipeline_cross_val(pipeline, X, y, labels=None, n_splits=3):
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
    kfold = IterativeStratification(n_splits=n_splits, order=1)
    folds = []
    pred = np.zeros_like(y, dtype=float)
    for i, (train_idx, valid_idx) in enumerate(kfold.split(X, y)):
        print(f"fold {i+1}/{kfold.n_splits}")
        folds.append((train_idx, valid_idx))
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        pipeline.fit(X_train, y_train)
        pred[valid_idx] = pipeline.predict(X_valid)
    mlc = MultiLabelClassification(y, pred=pred, labels=labels)
    return mlc, folds
