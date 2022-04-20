"""
Training script for genre classification pipelines
"""


import os
import sys
import pickle
import random
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml_utils import get_config, Logger, build_pipeline, plot_feature_importances


# Set random seeds for keras, numpy, and python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
RANDOM_SEED = 0
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Classifiers that can produce feature importance plots
FEATURE_IMPORTANCE_CLASSIFIERS = [
    'LogisticRegression',
    'RandomForestClassifier',
    'LGBMClassifier',
    'MultinomialNB',
]


if __name__ == '__main__':
    cfg = get_config()
    sys.stdout = Logger(os.path.join(cfg['output'], 'train.log'))
    df = pd.read_csv(cfg['input'])
    X = df.pop('lyrics').values
    y = df.values
    genres = df.columns
    print_header = f"Training models"
    print("\n\n" + "*" * len(print_header) + f"\n{print_header}\n" + "*" * len(print_header))
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of labels: {y.shape[1]}")
    print(f"Labels: {list(genres)}")
    print("Models:", [model['name'] for model in cfg['models']])
    for model in cfg['models']:
        name = model['name']
        if name == 'KerasClassifier':
            from ml_utils import init_tf_session
            init_tf_session(RANDOM_SEED)
            model['params']['input_dim'] = cfg['vectorizer']['params']['max_features']
            model['params']['output_dim'] = y.shape[1]
        subdir = os.path.join(cfg['output'], name)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        print_header = f"{name} training"
        print("\n\n" + "*" * len(print_header) + f"\n{print_header}\n" + "*" * len(print_header))
        print("\nModel parameters:", model['params'])
        pipeline = build_pipeline(cfg['vectorizer'], cfg['resampler'], model)
        pipeline.fit(X, y, labels=genres)
        if 'threshold' in model.keys():
            pipeline.set_threshold(model['threshold'])
        vocab = np.array(pipeline.vectorizer.get_feature_names())
        if model['name'] in FEATURE_IMPORTANCE_CLASSIFIERS:
            for genre, clf in zip(genres, pipeline.classifier.classifiers_):
                plot_feature_importances(clf, vocab)
                filename = os.path.join(subdir, f"{name}_features_{genre}.png")
                plt.savefig(filename, bbox_inches='tight')
        if name == 'KerasClassifier':
            # extra steps for saving Keras models because they can't be pickled
            clf = pipeline.classifier
            clf_filename = os.path.join(subdir, 'model.h5')
            clf.model.save(clf_filename)
            pipeline.classifier = None
            clf_h5 = h5py.File(clf_filename, 'a')
            clf_h5.attrs['classes'] = clf.classes_
        with open(os.path.join(subdir, 'pipeline.pkl'), 'wb') as f:
            pickle.dump(pipeline, f)