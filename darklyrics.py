#! /usr/bin/env python

import os
import pickle as pkl
from argparse import ArgumentParser
from traceback import print_exc

from darklyrics import get_band_lyrics

parser = ArgumentParser()
parser.add_argument("bands", nargs="*")
parser.add_argument("-f", "--file")
parser.add_argument("-o", "--outdir")
parser.add_argument("-v", "--verbose", action="store_true", default=False)
args = parser.parse_args()

bands = args.bands
file = args.file
outdir = args.outdir
verbose = args.verbose
if file is not None:
    with open(file) as f:
        bands = [line.rstrip() for line in f.readlines()]
for band in bands:
    if verbose:
        print(band)
    try:
        basename = band.lower().replace(' ', '')
        lyrics = get_band_lyrics(basename, verbose=verbose)
        if outdir is not None and len(lyrics) > 0:
            filename = os.path.join(outdir, basename + '.pkl')
            with open(filename, 'wb') as f:
                pkl.dump(lyrics, f, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            print(lyrics)
    except KeyboardInterrupt:
        raise
    except Exception:
        print_exc()
