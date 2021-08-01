import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lyrics_utils import get_config
from ml_utils import Logger, build_pipeline, multilabel_pipeline_cross_val


if __name__ == '__main__':
    cfg = get_config()
    sys.stdout = Logger(os.path.join(cfg['output'], 'grid_search.log'))
    df = pd.read_csv(cfg['input'])[::cfg['downsample']]
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
        est_params_lists = {}
        est_params_fixed = {}
        for k, v in est_params.items():
            if isinstance(v, list):
                est_params_lists[k] = v
            else:
                est_params_fixed[k] = v
        est_param_grid = list(itertools.product(*est_params_lists.values()))
        results = []
        for p in est_param_grid:
            print("\n\n********************\n"
                  f"{name} cross-validation\n"
                  f"********************\n")
            p = dict(zip(est_params_lists.keys(), p))
            p.update(est_params_fixed)
            print(p)
            print()
            pipeline = build_pipeline(vectorizer_params, mlsol_params, est, p)
            mlc, _ = multilabel_pipeline_cross_val(pipeline, X, y, labels=genres)
            mlc.print_report()
            best_thresh = mlc.thresh
            if isinstance(best_thresh, float):
                best_thresh = np.ones(mlc.n_labels) * best_thresh
            if len(best_thresh.shape) == 2:
                best_thresh = best_thresh.mean(0)
            print("\nDecision thresholds:")
            for i, label in enumerate(mlc.labels):
                print(f"  {label:<10s}: {best_thresh[i]:.2f}")
            results.append([mlc.f1_score, mlc.hamming_loss, np.mean(mlc.roc_auc_score())])
        print("\n\n********************\n"
              "Summary (f1_score, hamming_loss, macro-average roc_auc_score):\n")
        for p, res in zip(est_param_grid, results):
            print(f"{p}\t{res[0]:.3f}\t{res[1]:.3f}\t{res[2]:.3f}")
