import zlib
import base64
from io import StringIO
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

def compress_chunk(d):
    return compress_string(d.to_csv(None, sep=";", index=False))


def decompress_chunk(sz):
    s = decompress_string(sz)
    out = pd.DataFrame()
    if s != "\n":
        out = pd.read_csv(StringIO(s), sep=";")
        for k in ["timestamp", "context__tl"]:
            if k in out.columns:
                out[k] = pd.to_datetime(out[k])
    return out


def decompress_string(sz):
    return zlib.decompress(base64.b64decode(sz)).decode("utf8")


def compress_string(s):
    return base64.b64encode(zlib.compress(s.encode("utf8"))).decode("utf8")


def read_compressed_datafile(fname):
    with open(fname, "r") as fp:
        F_str = fp.read()
    F = decompress_chunk(F_str)
    return F


def write_compressed_datafile(F, fname):
    F_str = compress_chunk(F)
    with open(fname, "w") as fp:
        fp.write(F_str)


def main(args):
    infile = args.i
    F = read_compressed_datafile(infile)
    F.describe()

    # Each row of F contains a flattened 3d time series + context info
    X = F[[s for s in F.columns if "feats__" in s]].values
    Y = F[["target__y"]].values.reshape(-1, 1)

    x = X[0].reshape(-1, 3)
    y = Y[0]

    # x is an example of a 2Hz, 3d time series of length 60 min 
    # ndarray of size (n_samples, 3) 

    n_samples, d = x.shape
    sample_per_sec = 2
    time_sec = np.arange(n_samples) / sample_per_sec
    signames = ["btb", "rf", "spo2"]

    plt.figure()
    plt.plot(time_sec, x.reshape(-1, 3))
    plt.legend(signames)
    plt.title("Target:{}".format(y))
    plt.savefig("Example signals.pdf")
    plt.close()


parser = argparse.ArgumentParser()
parser.add_argument("-i", help="input *.dat file", type=str, required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    sys.exit(0)
