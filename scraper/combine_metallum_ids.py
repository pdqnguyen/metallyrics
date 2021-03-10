import os
from argparse import ArgumentParser
import pandas as pd


def main(inp, out):
    files = os.listdir(inp)
    df_list = []
    for f in files:
        path = os.path.join(inp, f)
        if '.csv' in path and path != out:
            df = pd.read_csv(path)
            df_list.append(df)
    df = pd.concat(df_list)
    df.to_csv(out, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input", help="directory containing metal-archives id lists")
    parser.add_argument("output", help="combined ids vilename")
    args = parser.parse_args()
    main(args.input, args.output)