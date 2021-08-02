import os
from argparse import ArgumentParser
import pickle
import warnings


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("lyrics", nargs='?')
    parser.add_argument("-i", "--interactive", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    print("Thresholds:", model.threshold)
    if args.lyrics is not None:
        model.classify_text(args.lyrics)
    else:
        while True:
            try:
                inp = input("\nEnter your lyrics:")
            except (KeyboardInterrupt, EOFError):
                break
            else:
                print()
                model.classify_text(inp)
