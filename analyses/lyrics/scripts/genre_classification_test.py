"""
Testing script for a genre classification pipeline
Runs in interactive mode if no lyrics are given in command-line
"""


import os
import glob
from argparse import ArgumentParser
import pickle
import warnings


if __name__ == '__main__':
    # handling of warning messages, gets rid of most things but not everything
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    parser = ArgumentParser()
    parser.add_argument("pipeline")
    parser.add_argument("lyrics", nargs='?')
    parser.add_argument("-k", "--keras", action="store_true", default=False)
    parser.add_argument("-t", "--thresholds")
    args = parser.parse_args()

    with open(args.pipeline, 'rb') as f:
        pipeline = pickle.load(f)
    if args.keras:
        # extra code for rebuilding a KerasClassifier wrapper object
        # this is necessary because KerasClassifiers are not well suited for I/O
        # but scikit-multilearn handles KerasClassifiers better than native Keras models
        import h5py
        from keras.wrappers.scikit_learn import KerasClassifier
        from keras.models import load_model
        from ml_utils import create_keras_model
        # look for and load all Keras models found in the same directory as the pipeline
        pipeline_dir = os.path.dirname(args.pipeline)
        classifiers = []
        clf_fnames = glob.glob(os.path.join(pipeline_dir, 'keras*.h5'))
        for i, clf_fname in enumerate(clf_fnames):
            clf = KerasClassifier(create_keras_model)
            clf.model = load_model(clf_fname)
            # get the 'classes_' attribute, necessary for the wrapper to make predictions
            with h5py.File(clf_fname, 'r') as clf_h5:
                clf.classes_ = clf_h5.attrs['classes']
            classifiers.append(clf)
        # make sure number of Keras models found matches the number needed in the pipeline
        if len(classifiers) == len(pipeline.classifier.classifiers_):
            pipeline.classifier.classifiers_ = classifiers
        else:
            raise OSError(f"{len(classifiers)} Keras models found,"
                          f" {len(pipeline.classifier.classifiers_)} needed")
    if args.thresholds:
        pipeline.set_threshold([float(tt) for tt in args.thresholds.split(',')])
    print("Thresholds:", pipeline.threshold)
    if args.lyrics is not None:
        pipeline.classify_text(args.lyrics, verbose=True)
    else:
        while True:
            try:
                inp = input("\nEnter your lyrics:")
            except (KeyboardInterrupt, EOFError):
                break
            else:
                print()
                pipeline.classify_text(inp, verbose=True)
