import os
import re
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold

from imblearn.over_sampling import RandomOverSampler, SMOTE

from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers, optimizers
from keras.callbacks import EarlyStopping
from keras.utils import Sequence

from multilabel import MultiLabelClassification


# DEFAULTS
DEFAULT_W2V_WV = 'glove-wiki-gigaword-300'
DEFAULT_CROSS_VAL_SPLITS = 3
DEFAULT_CONV_NB_FILTERS = None
DEFAULT_KERNEL_SIZE = None
DEFAULT_FULLY_CONNECTED_SIZE = None
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 1


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("-o", "--outdir")
    parser.add_argument("-w", "--word-vectors", default=DEFAULT_W2V_WV)
    parser.add_argument("-c", "--cross-val-splits", default=DEFAULT_CROSS_VAL_SPLITS)
    parser.add_argument("-n", "--conv-nb-filters", default=DEFAULT_CONV_NB_FILTERS,
                        nargs='+', type=int)
    parser.add_argument("-k", "--conv-kernel-size", default=DEFAULT_KERNEL_SIZE,
                        nargs='+', type=int)
    parser.add_argument("-f", "--fully-connected-size", default=DEFAULT_FULLY_CONNECTED_SIZE,
                        nargs='+', type=int)
    parser.add_argument("-e", "--epochs", default=DEFAULT_EPOCHS, type=int)
    parser.add_argument("-b", "--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument("-p", "--patience", default=DEFAULT_PATIENCE, type=int)
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    return parser.parse_args()


def preprocess_data(corpus_train, corpus_valid, word_vectors):
    maxlen = max([len(doc.split()) for doc in corpus_train])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus_train)
    X_train = tokenizer.texts_to_sequences(corpus_train)
    X_valid = tokenizer.texts_to_sequences(corpus_valid)
    X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')
    X_valid = pad_sequences(X_valid, maxlen=maxlen, padding='post')
    embedding_matrix = generate_word_embedding(tokenizer, word_vectors)
    return X_train, X_valid, embedding_matrix


def generate_word_embedding(tokenizer, word_vectors):
    vocab_size = len(tokenizer.word_index) + 1
    vector_dim = word_vectors.vector_size
    embedding_matrix = np.zeros((vocab_size, vector_dim))
    for word, word_idx in tokenizer.word_index.items():
        try:
            embedding_vector = word_vectors[word]
        except KeyError:
            pass
        else:
            embedding_matrix[word_idx] = embedding_vector
    return embedding_matrix


def create_keras_model(
        embedding_matrix,
        input_length,
        nb_classes=1,
        conv_nb_filters=None,
        conv_kernel_size=None,
        fc_size=None,
        learning_rate=None,
    ):
    keras_model = Sequential()
    keras_model.add(
        layers.Embedding(
            embedding_matrix.shape[0],
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=input_length,
            trainable=False
        )
    )
    if isinstance(conv_nb_filters, list):
        if len(conv_nb_filters) != len(conv_kernel_size):
            raise ValueError("n_conv_filters and conv_size must be same length")
        for nb_filters, kernel_size in zip(conv_nb_filters, conv_kernel_size):
            keras_model.add(layers.Conv1D(nb_filters, kernel_size, activation='relu'))
            keras_model.add(layers.MaxPooling1D(2))
        keras_model.add(layers.Flatten())
    if isinstance(fc_size, list):
        for fc_size_ in fc_size:
            keras_model.add(layers.Dense(fc_size_, activation='relu'))
    if conv_nb_filters is None and fc_size is None:
        raise ValueError("No convolutional or fully connected layers provided")
    keras_model.add(layers.Dense(nb_classes, activation='sigmoid'))
    optimizer = optimizers.Adam(lr=0.01)
    keras_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return keras_model


class BatchGenerator(Sequence):

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_X, batch_y


def main():
    args = parse_args()
    df = pd.read_csv('songs-ml-10pct.csv')
    X = df.pop('lyrics').values
    y = df.values
    genres = df.columns
    nb_classes = len(genres)
    print(f"number of songs: {X.shape[0]}")
    print(f"number of labels: {nb_classes}")
    print(f"labels: {list(genres)}")

    # Output directory
    outdir = args.outdir
    if outdir is None:
        outdir = time.strftime(r"cnn_output", time.localtime())
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Word2vec model
    print("Loading Word2Vec model")
    word_vectors = KeyedVectors.load(args.word_vectors)

    # Cross-validation
    n_splits = args.cross_val_splits
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    # CNN model
    model_params = dict(
        nb_classes=nb_classes,
        conv_nb_filters=args.conv_nb_filters,
        conv_kernel_size=args.conv_kernel_size,
        fc_size=args.fully_connected_size,
    )

    # Training
    epochs = args.epochs
    batch_size = args.batch_size
    early_stopping = EarlyStopping(
        monitor='val_loss', mode='min', patience=args.patience,
        restore_best_weights=True)

    scores = np.zeros((n_splits, nb_classes))
    confusion_matrices = np.zeros((n_splits, nb_classes, 2, 2))
    results = []
    print("Performing {}-fold cross-validation".format(n_splits))
    for i, (train_idx, valid_idx) in enumerate(kfold.split(X, y)):
        print("-----\nCV fold {}/{}".format(i + 1, n_splits))
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        X_train, X_valid, embedding_matrix = preprocess_data(X_train, X_valid, word_vectors)
        keras_model = create_keras_model(embedding_matrix, X_train.shape[1], **model_params)
        train_generator = BatchGenerator(X_train, y_train, batch_size)
        validation_generator = BatchGenerator(X_valid, y_valid, batch_size)
        keras_model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=[early_stopping],
        )
        y_pred = keras_model.predict(X_valid)
        y_pred_classes = y_pred.round().astype(int)
        for j in range(nb_classes):
            scores[i, j] = balanced_accuracy_score(y_valid[:, j], y_pred_classes[:, j])
            confusion_matrices[i, j] = confusion_matrix(y_valid[:, j], y_pred_classes[:, j])
        results.append((y_valid, y_pred))
    print("-----\nCross-validation complete")
    print("CV score: {:.2f}% +/- {:.2f}%".format(scores.mean()* 100, scores.std() * 200))
    print("Average confusion matrix:")
    print(confusion_matrices.mean(axis=0) / confusion_matrices.sum() * n_splits)
    true = np.concatenate([res[0] for res in results])
    pred = np.concatenate([res[1] for res in results])
    mlc = MultiLabelClassification(true, pred, labels=genres)
    mlc.print_report()
    fig = mlc.plot_roc_curve()
    fig.savefig(os.path.join(outdir, "roc.png"))
    fig = mlc.plot_precision_recall_curve()
    fig.savefig(os.path.join(outdir, "precision_recall.png"))
    auc_scores = mlc.roc_auc_score()
    print(auc_scores)
    print("AUC ROC score: {:.2f} +/- {:.2f}".format(np.mean(auc_scores), np.std(auc_scores)))
    best_thresholds = mlc.best_thresholds()
    print("Best thresholds:")
    for i, label in enumerate(mlc.labels):
        print(f"{label:<10s}: {best_thresholds[i]:.2f}")

    pred_classes = np.zeros(pred.shape).astype(int)
    for i in range(nb_classes):
        pred_classes[:, i] = (pred[:, i] > best_thresholds[i])

    mlc = MultiLabelClassification(true, pred, y_pred_classes=pred_classes, labels=genres)
    mlc.print_report()
    scores = np.zeros((n_splits, nb_classes))
    confusion_matrices = np.zeros((n_splits, nb_classes, 2, 2))
    for i, (train_idx, valid_idx) in enumerate(kfold.split(X, y)):
        for j in range(nb_classes):
            scores[i, j] = balanced_accuracy_score(true[valid_idx, j], pred_classes[valid_idx, j])
            confusion_matrices[i, j] = confusion_matrix(true[valid_idx, j], pred_classes[valid_idx, j])
    print("-----\nCross-validation results using best thresholds")
    print("CV score: {:.2f}% +/- {:.2f}%".format(scores.mean() * 100, scores.std() * 200))
    print("Average confusion matrix:")
    print(confusion_matrices.mean(axis=0) / confusion_matrices.sum() * n_splits)
    return


if __name__ == '__main__':
    main()