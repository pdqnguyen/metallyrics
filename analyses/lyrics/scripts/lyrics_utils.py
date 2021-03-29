from argparse import ArgumentParser
import yaml
import pandas as pd


def get_config():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    for key in ['input', 'output']:
        if key not in cfg.keys():
            raise KeyError(f"missing field {key} in {args.config}")
    return cfg


def load_songs(filepath):
    data = pd.read_csv(filepath)
    data['song_words'] = data['song_words'].str.split(' ')
    return data
