import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lyrics_utils import get_config
from ml_utils import Logger, build_pipeline, plot_feature_importances


if __name__ == '__main__':
    cfg = get_config()
    sys.stdout = Logger(os.path.join(cfg['output'], 'train.log'))
    df = pd.read_csv(cfg['input'])
    X = df.pop('lyrics').values
    y = df.values
    genres = df.columns
    print(f"number of songs: {X.shape[0]}")
    print(f"number of labels: {y.shape[1]}")
    print(f"labels: {list(genres)}")
    vectorizer_params = cfg['vectorizer']
    mlsol_params = cfg['mlsol']
    for est_params in cfg['models']:
        name = est_params.pop('name')
        module = est_params.pop('module')
        exec(f"from {module} import {name}")
        est = eval(name)
        print("\n\n********************\n"
              f"{name}\n"
              f"********************\n")
        print(est_params)
        print()
        pipeline = build_pipeline(vectorizer_params, mlsol_params, est, est_params)
        pipeline.fit(X, y, label_names=genres)
        vocab = np.array(pipeline.vectorizer.get_feature_names())
        if name in ('RandomForestClassifier', 'LGBMClassifier'):
            for genre, clf in zip(genres, pipeline.classifier.classifiers_):
                plot_feature_importances(clf, vocab)
                filename = os.path.join(cfg['output'], f"{name}_features_{genre}.png")
                plt.savefig(filename, bbox_inches='tight')
        with open(os.path.join(cfg['output'], name + '.pkl'), 'wb') as f:
            pickle.dump(pipeline, f)
