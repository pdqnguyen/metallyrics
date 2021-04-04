"""
Compute a bunch of metrics for quantifying the complexity of song/artist lyrics
"""

import os
import random
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

import lyrics_utils as utils
import nlp


def TTR(x):
    return len(set(x)) / len(x)


def cumulative_TTR(words):
    out = [TTR(words[: i + 1]) for i in range(len(words))]
    return out


def MTLD_forward(words, threshold):
    factor = 0
    segment = []
    i = 0
    seg_ttr = 0
    while i < len(words):
        segment.append(words[i])
        seg_ttr = TTR(segment)
        if seg_ttr <= threshold:
            segment = []
            factor += 1
        i += 1
    if len(segment) > 0:
        factor += (1.0 - seg_ttr) / (1.0 - threshold)
    factor = max(1.0, factor)
    mtld = len(words) / factor
    return mtld


def MTLD(words, threshold=0.720):
    if len(words) == 0:
        return 0.0
    forward = MTLD_forward(words, threshold)
    reverse = MTLD_forward(words[::-1], threshold)
    return 0.5 * (forward + reverse)


def vocd_curve(x, d):
    return (d / x) * (np.sqrt(1 + 2 * (x / d)) - 1)


def vocd(words, num_trials=3, num_segs=100, min_seg=35, max_seg=50):
    if max_seg > len(words):
        return np.nan
    d_trials = []
    seglen_range = range(min_seg, max_seg + 1)
    for _ in range(num_trials):
        ttrs = np.zeros((len(seglen_range), num_segs))
        for i, seglen in enumerate(seglen_range):
            for j in range(num_segs):
                sample = random.sample(words, seglen)
                ttrs[i, j] = TTR(sample)
        avg_ttrs = ttrs.mean(1)
        (d_trial,), _ = curve_fit(vocd_curve, seglen_range, avg_ttrs)
        d_trials.append(d_trial)
    return np.mean(d_trials)


def get_lexical_diversity(data, measures):
    tokens = data.words.apply(len)
    types = data.words.apply(lambda x: len(set(x)))
    data['N'] = tokens
    if measures.get('types', False):
        data['types'] = types
    if measures.get('TTR', False):
        data['TTR'] = types / tokens
    if measures.get('logTTR', False):
        data['logTTR'] = np.log(types) / np.log(tokens)
    if measures.get('MTLD', False):
        data['MTLD'] = data.words.apply(MTLD, threshold=0.5)
    if measures.get('logMTLD', False):
        if 'MTLD' in data.columns:
            data['logMTLD'] = np.log(data['MTLD'])
        else:
            data['logMTLD'] = np.log(data.words.apply(MTLD, threshold=0.5))
    if measures.get('vocd-D', False):
        data['vocd-D'] = data.words.apply(vocd, num_segs=10)
    if measures.get('logvocd-D', False):
        if 'vocd-D' in data.columns:
            data['logvocd-D'] = np.log(data['vocd-D'])
        else:
            data['logvocd-D'] = np.log(data.words.apply(vocd, num_segs=10))
    return data[data.N > 0]


cfg = utils.get_config()
output = cfg['output']
if not os.path.exists(output):
    os.mkdir(output)
band_df = utils.load_bands(cfg['input'])
band_df = nlp.get_band_stats(band_df)
num_bands = cfg['num_bands']
top_reviewed = band_df.sort_values('reviews')['name'][-num_bands:]
band_df = band_df.loc[top_reviewed.index]
genres = [c for c in band_df.columns if 'genre_' in c]
ld_df = get_lexical_diversity(band_df, cfg['measures'])
filepath = os.path.join(cfg['output'], cfg['filename'])
ld_df.to_csv(filepath, index=False)

# for i, row in band_df.iterrows():
#     plt.plot(cumulative_TTR(row['words'][:2000]), alpha=0.3)
# plt.xlabel('Tokens')
# plt.ylabel('Cumulative TTR')
# plt.savefig(os.path.join(cfg['output'], 'TTR.png'))
