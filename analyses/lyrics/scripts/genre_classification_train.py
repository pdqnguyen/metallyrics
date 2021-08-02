import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml_utils import get_config, Logger, build_pipeline, plot_feature_importances


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
RANDOM_SEED = 0
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


FEATURE_IMPORTANCE_CLASSIFIERS = [
    'LogisticRegression',
    'RandomForestClassifier',
    'LGBMClassifier'
]


if __name__ == '__main__':
    cfg = get_config()
    sys.stdout = Logger(os.path.join(cfg['output'], 'train.log'))
    df = pd.read_csv(cfg['input'])[::10]
    X = df.pop('lyrics').values
    y = df.values
    genres = df.columns
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of labels: {y.shape[1]}")
    print(f"Labels: {list(genres)}")
    print("Models:", [model['name'] for model in cfg['models']])
    for model in cfg['models']:
        name = model['name']
        if name == 'KerasClassifier':
            from ml_utils import init_tf_session, get_vector_dim
            init_tf_session(RANDOM_SEED)
            model['params']['input_dim'] = get_vector_dim(X, cfg['vectorizer'])
        subdir = os.path.join(cfg['output'], name)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        print_header = f"{name} training"
        print("\n\n" + "*" * len(print_header) + f"\n{print_header}\n" + "*" * len(print_header))
        print("\nModel parameters:", model['params'])
        pipeline = build_pipeline(cfg['vectorizer'], cfg['resampler'], model)
        pipeline.fit(X, y, labels=genres)
        vocab = np.array(pipeline.vectorizer.get_feature_names())
        if model['name'] in FEATURE_IMPORTANCE_CLASSIFIERS:
            for genre, clf in zip(genres, pipeline.classifier.classifiers_):
                plot_feature_importances(clf, vocab)
                filename = os.path.join(cfg['output'], f"{name}_features_{genre}.png")
                plt.savefig(filename, bbox_inches='tight')
        with open(os.path.join(subdir, 'model.pkl'), 'wb') as f:
            pickle.dump(pipeline, f)
