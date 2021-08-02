import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lyrics_utils import get_config
from ml_utils import Logger, build_pipeline, multilabel_pipeline_cross_val


if __name__ == '__main__':
    cfg = get_config()
    sys.stdout = Logger(os.path.join(cfg['output'], 'cross_val.log'))
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
              f"{name} cross-validation\n"
              f"********************\n")
        print(est_params)
        print()
        pipeline = build_pipeline(vectorizer_params, mlsol_params, est, est_params)
        mlc = multilabel_pipeline_cross_val(pipeline, X[::10], y[::10], labels=genres, verbose=1)
        mlc.print_report()
        best_thresh = mlc.thresh
        if isinstance(best_thresh, float):
            best_thresh = np.ones(mlc.n_labels) * best_thresh
        if len(best_thresh.shape) == 2:
            best_thresh = best_thresh.mean(0)
        print("\nDecision thresholds:")
        for i, label in enumerate(mlc.labels):
            print(f"  {label:<10s}: {best_thresh[i]:.2f}")
        mlc.plot_roc_curve()
        plt.savefig(os.path.join(cfg['output'], name + '_roc.png'))
        plt.close()
        mlc.plot_precision_recall_curve()
        plt.savefig(os.path.join(cfg['output'], name + '_precn_recall.png'))
        plt.close()
