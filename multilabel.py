import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
sns.set(font_scale=2)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, multilabel_confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve
)


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
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
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
                print("CV scores:")
                print("Accuracy:    {:.2f} +/- {:.2f}".format(avg_accuracy, std_accuracy))
                print("Precision:   {:.2f} +/- {:.2f}".format(avg_precision, std_precision))
                print("Recall:      {:.2f} +/- {:.2f}".format(avg_recall, std_recall))
                print("F1-score:    {:.2f} +/- {:.2f}".format(avg_f1, std_f1))
                print("Average confusion matrix:")
                print(confusion_matrices.mean(axis=0) / confusion_matrices.sum() * n_splits)
        return MultiLabelClassification(y_valid, y_prob, labels=self.labels)


class MultiLabelClassification:

    def __init__(self, y_true, y_pred=None, y_pred_classes=None, labels=None, class_thresh=0.5):
        self.true = y_true.astype(int)
        if y_pred_classes is None and y_pred is not None:
            self.pred = y_pred
            y_pred_classes = np.zeros_like(self.pred, dtype=int)
            y_pred_classes[self.pred > class_thresh] = 1
        else:
            self.pred = None
        self.pred_classes = y_pred_classes
        self.n_samples, self.n_labels = y_true.shape
        if labels is not None:
            if len(labels) == self.n_labels:
                self.labels = np.array(labels)
            else:
                raise ValueError("length of labels and shape of y_true do not match")
        else:
            self.labels = np.arange(self.true.shape[1])

    @property
    def __intersection(self):
        return self.true * self.pred_classes

    @property
    def __union(self):
        return np.minimum(1, self.true + self.pred_classes)

    @property
    def accuracy_score(self):
        # Number of labels in common / overall labels (true and predicted)
        return np.nanmean(self.__intersection.sum(1) / self.__union.sum(1))

    @property
    def precision_score(self):
        # Proportion of predicted labels that are correct
        return np.nanmean(self.__intersection.sum(1) / self.pred_classes.sum(1))

    @property
    def recall_score(self):
        # Proportion of true labels that were predicted
        return np.nanmean(self.__intersection.sum(1) / self.true.sum(1))

    @property
    def f1_score(self):
        # Harmonic mean of precision_score and recall_score
        p = self.precision_score
        r = self.recall_score
        return 2 * (p * r) / (p + r)

    @property
    def hamming_loss(self):
        # Symmetric difference b/w pred and true labels (true XOR pred)
        delta = np.zeros(self.true.shape[0])
        for i in range(delta.shape[0]):
            delta[i] = np.sum(self.true[i] ^ self.pred_classes[i])
        return delta.mean()

    def confusion_matrix(self, label=None, label_idx=None):
        confusion_matrices = multilabel_confusion_matrix(self.true, self.pred_classes)
        if label is not None:
            return confusion_matrices[np.where(self.labels == label)[0][0]]
        elif label_idx is not None:
            return confusion_matrices[label_idx]
        else:
            return confusion_matrices

    def print_report(self, verbose=0):
        print("Multi-label classification report:")
        print("Accuracy:      {:.2f}".format(self.accuracy_score))
        print("Precision:     {:.2f}".format(self.precision_score))
        print("Recall:        {:.2f}".format(self.recall_score))
        print("F1-score:      {:.2f}".format(self.f1_score))
        print("Hamming loss:  {:.2f}".format(self.hamming_loss))
        if verbose == 1:
            for label, matrix in zip(self.labels, self.confusion_matrix()):
                print("===\nLabel: {}".format(label))
                print(matrix)
        return

    def roc_auc_score(self):
        auc = np.zeros(len(self.labels))
        for i, label in enumerate(self.labels):
            auc[i] = roc_auc_score(self.true[:, i], self.pred[:, i])
        return auc

    def best_thresholds(self):
        best = np.zeros(len(self.labels))
        for i, label in enumerate(self.labels):
            fpr, tpr, thresholds = roc_curve(self.true[:, i], self.pred[:, i])
            gmean = np.sqrt(tpr * (1 - fpr))
            best[i] = thresholds[gmean.argmax()]
        return best

    def plot_roc_curve(self):
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
        return fig

    def plot_precision_recall_curve(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for i, label in enumerate(self.labels):
            precision, recall, thresholds = precision_recall_curve(self.true[:, i], self.pred[:, i])
            line = ax.plot(thresholds, precision[:-1], label=label)
            ax.plot(thresholds, recall[:-1], ":", color=line[0].get_color())
        ax.set_aspect('equal')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title("Precision and recall scores\nvs decision threshold")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Score")
        ax.text(0.01, 0.01, "solid lines show precision score\ndotted lines show recall score", size=16)
        ax.legend(loc='upper right')
        ax.grid(True)
        return fig
