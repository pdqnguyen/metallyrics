import os
import sys
import itertools
import random
import numpy as np
import pandas as pd

from ml_utils import get_config, Logger, build_pipeline, multilabel_pipeline_cross_val


# Set random seeds for keras, numpy, and python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
RANDOM_SEED = 0
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


if __name__ == '__main__':
    cfg = get_config()
    df = pd.read_csv(cfg['input'])[::cfg['downsample']]
    X = df.pop('lyrics').values
    y = df.values
    genres = df.columns
    print_header = f"Hyperparameter tuning"
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
            # Special handling for Keras models.
            # A build function must be provided as a positional argument
            # The feature matrix must also be padded to the max_features so that
            # the input_dim parameter can be chosen agnostic of the dimensions of
            # the un-padded vectorized feature matrix.
            model['params']['input_dim'] = cfg['vectorizer']['params']['max_features']
            # The output dimension is also provided because the Keras model is a
            # multi-label classifier on its own, whereas other models are single-label
            # classifiers fed into a Binary Relevance metamodel
            model['params']['output_dim'] = y.shape[1]
        subdir = os.path.join(cfg['output'], name)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        sys.stdout = Logger(os.path.join(subdir, 'grid_search.log'))
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
        thresholds = []
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
            mlc = multilabel_pipeline_cross_val(
                pipeline,
                X,
                y,
                labels=genres,
                n_splits=cfg['n_splits'],
                verbose=2
            )
            roc_fig = mlc.plot_roc_curve()
            roc_fig.savefig(os.path.join(subdir, "roc.png"))
            results.append([mlc.f1_score, mlc.hamming_loss, np.mean(mlc.roc_auc_score)])
            thresholds.append(mlc.threshold)
        mlc.to_csv(os.path.join(subdir, "results.csv"))
        with open(os.path.join(subdir, "summary.txt"), 'w') as f:
            header = "Summary (hyperparameters, f1_score, hamming_loss, macro-average roc_auc_score," \
                     " best thresholds):\n"
            f.write(header)
            print(header)
            for p, res, thresh in zip(params_grid, results, thresholds):
                row = f"{p}{res[0]:>10.3f}{res[1]:>10.3f}{res[2]:>10.3f}     {thresh}"
                f.write(row + "\n")
                print(row)
