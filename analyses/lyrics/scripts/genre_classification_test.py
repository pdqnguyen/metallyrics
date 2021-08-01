from argparse import ArgumentParser
import pickle
import numpy as np


thresh = np.array([0.64, 0.69, 0.65, 0.65])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("lyrics")
    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    model.classify_text(args.lyrics)
