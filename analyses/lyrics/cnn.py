import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from gensim.models import KeyedVectors

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers, optimizers
from keras.callbacks import EarlyStopping
from keras.utils import Sequence
# from keras.constraints import maxnorm

from skmultilearn.model_selection import IterativeStratification

from mlsol import MLSOL
from multilabel import MultiLabelClassification, multilabel_cross_val_report


# DEFAULTS
DEFAULT_CROSS_VAL_SPLITS = 3
DEFAULT_CONV_NB_FILTERS = None
DEFAULT_KERNEL_SIZE = None
DEFAULT_FULLY_CONNECTED_SIZE = None
DEFAULT_BATCH_NORMALIZATION = False
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 1
DEFAULT_OVERSAMPLE = False
DEFAULT_W2V_WV = 'glove-wiki-gigaword-300'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("-o", "--outdir")
    parser.add_argument(
        "-w", "--word-vectors", default=DEFAULT_W2V_WV)
    parser.add_argument(
        "-c", "--cross-val-splits", default=DEFAULT_CROSS_VAL_SPLITS,
        type=int)
    parser.add_argument(
        "-n", "--conv-nb-filters", default=DEFAULT_CONV_NB_FILTERS,
        nargs='+', type=int)
    parser.add_argument(
        "-k", "--conv-kernel-size", default=DEFAULT_KERNEL_SIZE,
        nargs='+', type=int)
    parser.add_argument(
        "-f", "--fully-connected-size",
        default=DEFAULT_FULLY_CONNECTED_SIZE, nargs='+', type=int)
    parser.add_argument(
        "-t", "--batch-normalization",
        default=DEFAULT_BATCH_NORMALIZATION, action="store_true")
    parser.add_argument(
        "-l", "--learning-rate", default=DEFAULT_LEARNING_RATE, type=float)
    parser.add_argument(
        "-e", "--epochs", default=DEFAULT_EPOCHS, type=int)
    parser.add_argument(
        "-b", "--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument(
        "-p", "--patience", default=DEFAULT_PATIENCE, type=int)
    parser.add_argument(
        "-s", "--oversample", default=DEFAULT_OVERSAMPLE, action="store_true")
    parser.add_argument(
        "-v", "--verbose", default=False, action="store_true")
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


def create_model(
        embedding_matrix,
        input_length,
        nb_classes=1,
        conv_nb_filters=DEFAULT_CONV_NB_FILTERS,
        conv_kernel_size=DEFAULT_KERNEL_SIZE,
        fc_size=DEFAULT_FULLY_CONNECTED_SIZE,
        batch_normalization=DEFAULT_BATCH_NORMALIZATION,
        learning_rate=DEFAULT_LEARNING_RATE,
):
    model = Sequential()
    model.add(
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
            raise ValueError(
                "conv_nb_filters and conv_kernel_size must be same length")
        conv_layers = zip(conv_nb_filters, conv_kernel_size)
        for nb_filters, kernel_size in conv_layers:
            model.add(layers.Conv1D(nb_filters, kernel_size))
            if batch_normalization:
                model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            model.add(layers.MaxPooling1D(2))
        model.add(layers.Flatten())
    if isinstance(fc_size, list):
        for fc_size_ in fc_size:
            model.add(layers.Dense(fc_size_, activation='relu'))
    if conv_nb_filters is None and fc_size is None:
        raise ValueError("neither conv_nb_filters or fc_size provided")
    model.add(layers.Dense(nb_classes, activation='sigmoid'))
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    return model


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


def cross_validate(
        X,
        y,
        word_vectors,
        labels=None,
        n_splits=DEFAULT_CROSS_VAL_SPLITS,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        callbacks=None,
        oversample=DEFAULT_OVERSAMPLE,
        logfile=None,
        **model_params
):
    kfold = IterativeStratification(n_splits=n_splits, order=1, random_state=0)
    results = []
    folds = []
    print("Performing {}-fold cross-validation".format(n_splits))
    for i, (train_idx, valid_idx) in enumerate(kfold.split(X, y)):
        print("-----\nCV fold {}/{}".format(i + 1, n_splits))
        folds.append((train_idx, valid_idx))
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        X_train, X_valid, embedding_matrix = preprocess_data(
            X_train, X_valid, word_vectors)
        if oversample:
            print("Performing multi-label oversampling")
            mlsol = MLSOL(perc_gen_instances=0.3, k=5)
            X_train, y_train = mlsol.fit_resample(X_train, y_train)
        model = create_model(
            embedding_matrix, X_train.shape[1], **model_params)
        if i == 0 and logfile is not None:
            with open(logfile, 'w') as file:
                model.summary(print_fn=lambda x: file.write(x + '\n'))
                file.write(
                    "Training parameters:\n"
                    f"optimizer: {model.optimizer}\n"
                    f"learning rate: {model.optimizer.lr}\n"
                    f"epochs: {epochs}\n"
                    f"batch_size: {batch_size}\n"
                    f"oversample: {oversample}\n"
                )
                for callback in callbacks:
                    if isinstance(callback, EarlyStopping):
                        file.write(
                            f"early stopping monitor: {callback.monitor}\n"
                            f"early stopping min_delta: {callback.min_delta}\n"
                            f"early stopping patience: {callback.patience}\n"
                            f"early stopping wait: {callback.wait}\n"
                        )
        train_generator = BatchGenerator(X_train, y_train, batch_size)
        validation_generator = BatchGenerator(X_valid, y_valid, batch_size)
        model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
        )
        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_valid)
        # Optimize decision threshold
        mlc_train = MultiLabelClassification(
            y_train, y_train_pred, labels=labels)
        thresh = mlc_train.best_thresholds(metric='fscore', fbeta=1)
        print(thresh)
        thresh_r = np.tile(thresh, (y_pred.shape[0], 1))
        results.append((y_valid, y_pred, thresh_r))
    true = np.concatenate([res[0] for res in results])
    pred = np.concatenate([res[1] for res in results])
    thresh = np.concatenate([res[2] for res in results])
    mlc = MultiLabelClassification(true, pred, labels=labels, thresh=thresh)
    return mlc, folds


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

    # CNN model parameters
    model_params = dict(
        nb_classes=nb_classes,
        conv_nb_filters=args.conv_nb_filters,
        conv_kernel_size=args.conv_kernel_size,
        fc_size=args.fully_connected_size,
        batch_normalization=args.batch_normalization,
        learning_rate=args.learning_rate,
    )

    # Cross-validation of Keras model
    n_splits = args.cross_val_splits
    mlc, folds = cross_validate(
        X,
        y,
        word_vectors,
        labels=genres,
        n_splits=n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=args.patience,
            restore_best_weights=True
        )],
        oversample=args.oversample,
        logfile=os.path.join(outdir, "model.txt"),
        **model_params
    )

    # Report results with baseline decision threshold = 0.5
    print("-----\n\nCross-validation results:")
    metrics = [
        'balanced_accuracy_score', 'precision_score',
        'recall_score', 'f1_score']
    mlc.print_report()
    stdout = sys.stdout
    report_filename = os.path.join(outdir, 'cross_val_report.txt')
    if os.path.exists(report_filename):
        os.remove(report_filename)
    sys.stdout = open(report_filename, 'a')
    mlc.print_report()
    sys.stdout = stdout

    # Save performance plots and classification results
    fig = mlc.plot_roc_curve()
    fig.savefig(os.path.join(outdir, "roc.png"))
    fig = mlc.plot_precision_recall_curve()
    fig.savefig(os.path.join(outdir, "precision_recall.png"))
    mlc.to_csv(os.path.join(outdir, "results.csv"))
    for i, (train_idx, valid_idx) in enumerate(folds):
        mlc_fold = MultiLabelClassification(
            mlc.true[valid_idx, :],
            mlc.pred[valid_idx, :],
            mlc.pred_class[valid_idx, :],
            labels=mlc.labels,
        )
        mlc_fold.to_csv(os.path.join(outdir, f"results_fold_{i}.csv"))
    return


if __name__ == '__main__':
    main()
