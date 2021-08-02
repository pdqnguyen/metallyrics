import os
import sys
import itertools
import numpy as np
import pandas as pd

from ml_utils import get_config, Logger, build_pipeline, multilabel_pipeline_cross_val


if __name__ == '__main__':
    cfg = get_config()
    sys.stdout = Logger(os.path.join(cfg['output'], 'grid_search.log'))
    df = pd.read_csv(cfg['input'])[::cfg['downsample']]
    X = df.pop('lyrics').values
    y = df.values
    genres = df.columns
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of labels: {y.shape[1]}")
    print(f"Labels: {list(genres)}")
    print("Models:", [model['name'] for model in cfg['models']])
    for model in cfg['models']:
        name = model['name']
        subdir = os.path.join(cfg['output'], name)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        # Split parameters into iterable and non-iterable parameters
        params_lists = {}
        params_fixed = {}
        for k, v in model['params'].items():
            if isinstance(v, list):
                params_lists[k] = v
            else:
                params_fixed[k] = v
        # Create combinations of iterable parameters
        params_grid = list(itertools.product(*params_lists.values()))
        results = []
        print_header = f"{name} grid search"
        print("\n\n" + "*" * len(print_header) + f"\n{print_header}\n" + "*" * len(print_header))
        for p in params_grid:
            # Make a new "config dict" for this model using this set parameters
            # by combining the current variable parameters with the fixed ones
            p_dict = dict(zip(params_lists.keys(), p))
            p_dict.update(params_fixed)
            print("\nModel parameters:", p_dict)
            est_cfg = model.copy()
            est_cfg['params'].update(p_dict)
            # Build and cross-validate pipeline
            pipeline = build_pipeline(cfg['vectorizer'], cfg['resampler'], est_cfg)
            mlc = multilabel_pipeline_cross_val(pipeline, X, y, labels=genres, verbose=1)
            results.append([mlc.f1_score, mlc.hamming_loss, np.mean(mlc.roc_auc_score)])
        print("Summary (f1_score, hamming_loss, macro-average roc_auc_score):\n")
        for p, res in zip(params_grid, results):
            print(f"{p}{res[0]:>10.3f}{res[1]:>10.3f}{res[2]:>10.3f}")
