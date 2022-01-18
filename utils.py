"""
Author: Antoine Honoré
Contact: antoine.honore@ki.se
"""

import sys
from sqlalchemy import create_engine
import base64
import pandas as pd
from io import StringIO
import zlib
import os
# Got module not found error here. Solved by copying the pidprint-function from Carolins utils-file
#from patdb_tbox.print.utils import pidprint
import socket
from datetime import datetime
import numpy as np


def decompress_hf(s: str):
    """Decompress hf data which are different from lf data.

    - No pd.DataFrame format
    - no datetime data
    - sampling frequency as the first sample.

    """
    x = np.array(list(map(float, decompress_string(s).split(";"))))
    return x


def apply__bw_lim(s: str, opts_str: str) -> str:
    """Replace flags $bw_pop_low$ $bw_pop_high$ with values defined in opts_str.
    opts_str should match '^(lt[0-9]*|gt[0-9]*|bt[0-9]*\.[0-9])$'
    """
    bw_lim_low = "0"
    bw_lim_high = "20000"
    if opts_str != "na":
        if opts_str[:2] == "lt":
            bw_lim_high = opts_str[2:]
        elif opts_str[:2] == "gt":
            bw_lim_low = opts_str[2:]
        elif opts_str[:2] == "bt":
            bw_lim_low, bw_lim_high = opts_str[2:].split(".")
        else:
            pidprint("Unknown bw_lim parameter:", opts_str, "Exit.", flag="error")
            sys.exit(1)
        sout = s.replace("$bw_pop_low$", bw_lim_low).replace("$bw_pop_high$", bw_lim_high)

    else:
        sout = s
    return sout


def check_exp(exp_name, engine=None, ref_count=20):
    with engine.connect() as con:
        df = pd.read_sql("select * from results.res_tbl_{}".format(exp_name), con=con)
    all_cols = [s for s in list(df) if (s.startswith("data_opts__") or s.startswith("model_opts__"))] + ["match"]
    nunique = [df[c].unique() for c in all_cols]
    fixed_cols = [(c, "{}: {}".format(c, i[0])) for i, c in zip(nunique, all_cols) if i.shape[0] == 1]
    varying_cols = [(c, "{}: {}".format(c, i)) for i, c in zip(nunique, all_cols) if i.shape[0] > 1]
    sout = "\n---Fixed conditions\n"
    sout += "\n".join([f[1] for f in fixed_cols])

    sout += "\n\n---Varying conditions\n"
    sout += "\n".join([f[1] for f in varying_cols])

    thedf = df.drop(columns=[f[0] for f in fixed_cols] + ["idx"])

    dfcount = thedf.groupby([f[0] for f in varying_cols]).apply(lambda dd: dd.count())["test__AUROC"]
    sout += "\n\n---Runs count != {}\n".format(ref_count)
    sout += dfcount[dfcount != ref_count].to_string()

    return sout


def get_demographics(engine, period):
    """Get the data from `viewers.uid_demos_$period$` database view."""
    with engine.connect() as con:
        out = pd.read_sql("select * from viewers.uid_demos_{}".format(period), con).drop_duplicates(subset="ids__uid")
    return out


def format_query(opts_glob: dict, opts_data: dict, plim_override=None, period=None) -> str:
    """
    Replace keys in the template queries with data specified in `opts_glob` and `opts_data`.
    Limit the number of queried patients is the host is my laptop for debug purpuses.

    Inputs:

    - opts_glob: dict,
        - 'template': template query filename wo '.sql'
        - 'key1': first patient group to pull, if 'template' == "dl_negation" | "dl_binary"
        - 'key2': second patient group to pull if 'template' == "dl_binary"
    - opts_data: dict,
        - 'plim' if plim_override is None
        - 'bw_pop' (see `patdb_tbox.psql.psql.apply__bw_lim)
    - plim_override: override opts_data['plim'] argument

    """
    query_fname = opts_glob["template"] + ".sql"

    if not (plim_override is None):
        query_str = read_query_file(query_fname, period=period) \
            .replace("$limit$", "limit {}".format(plim_override))
    else:
        query_str = read_query_file(query_fname, period=period) \
            .replace("$limit$", "" if opts_data["plim"] == 0 else "limit {}".format(opts_data["plim"]))

    if "dl_negation" in opts_glob["template"]:
        # if we are defining the positive class
        # Replace the key with the event passed in the parameters
        query_str = query_str.replace("$key$", opts_glob["key1"])

    elif "dl_binary" in opts_glob["template"]:
        query_str = query_str.replace("$key1$", opts_glob["key1"]).replace("$key2$", opts_glob["key2"])
    else:
        pass

    query_str = apply__bw_lim(query_str, opts_data["bw_pop"])

    if socket.gethostname() == "cmm0576":
        query_str = query_str.replace("--limit", "limit")

    return query_str


def run_query(s: str, engine, verbose=False) -> pd.DataFrame:
    """
    Runs the query specified as a string vs a db engine.

    Inputs:

    - s:str, query
    - engine: (see patdb_tbox.psql.psql.create_engine)

    Returns:

        - pd.DataFrame
    """
    if verbose:
        pidprint("\n", s, flag="info")
    start_dl_time = datetime.now()
    df = pd.read_sql(s, engine)
    end_dl_time = datetime.now()
    dl_time = (end_dl_time - start_dl_time).total_seconds()
    memusage_MB = df.memory_usage(index=True, deep=True).sum() / 1024 / 1024

    if verbose:
        pidprint("dl_time={} sec, volume={} MB, link speed={} MB/s".format(round(dl_time, 3), round(memusage_MB, 3),
                                                                           round(memusage_MB / dl_time, 3)),
                 flag="report")
    return df


def read_passwd(username: str = "remotedbuser", root_folder: str = ".") -> str:
    """
    Read `username` password file from the `root_folder`.
    """
    with open(os.path.join(root_folder, "{}_dbfile.txt".format(username)), "r") as f:
        s = f.read().strip()
    return s


def get_engine(username: str = "remotedbuser", root_folder: str = ".", nodename: str = "client", schema=None, dbname:str="remotedb", verbose=False):
    """
    Get a database `sqlalchemy.engine` object for the user `username`, using ssl certificates specific for 'nodenames' type machines.
    For details about the database engine object see `sqlalchemy.create_engine`
    """

    passwd = read_passwd(username=username, root_folder=root_folder)
    connect_args = {}
    if username == "remotedbdata":
        connect_args = {'sslrootcert': os.path.join(root_folder, "root.crt"),
                        'sslcert': os.path.join(root_folder, "{}.crt".format(nodename)),
                        'sslkey': os.path.join(root_folder, "{}.key".format(nodename))}

    engine = create_engine('postgresql://{}:{}@127.0.0.1:5432/{}'.format(username, passwd, dbname),
                           connect_args=connect_args)
    with engine.connect() as con:
        if verbose:
            pidprint("Connection OK", flag="report")
        else:
            pass
    return engine


def read_query_file(fname: str, period: str = "weeks") -> str:
    """
    Read the template data query files.
    Replace the 'period' keyword in the template the `period` kwarg
    """
    with open(fname, "rb") as fp:
        query_str = fp.read().decode("utf8").replace("$period$", period)
    return query_str


def compress_chunks(D):
    """
    **obselete**
    """
    out = {"dummy": [compress_chunk(pd.DataFrame())] for _ in range(len(D))}
    if not all(d.empty for d in D):
        names = [list(d)[-1] for d in D]
        out = {k: [v] for (k, v) in zip(names, list(map(compress_chunk, D)))}
    return out


def compress_chunk(d: pd.DataFrame):
    """
    **Obselete** compress a dataframe.
    """
    return compress_string(d.to_csv(None, sep=";", index=False))


def decompress_chunk(sz: str, verbose=False, nrows=None):
    """
    **Obselete** decompress a dataframe.
    """
    s = decompress_string(sz, verbose=verbose)

    out = pd.DataFrame()
    if s != "\n":
        if verbose:
            pidprint("str2DataFrame...")

        out = pd.read_csv(StringIO(s), sep=";", nrows=nrows)

        if verbose:
            pidprint("Parse time columns...")
        for k in ["timestamp", "context__tl", "date"]:
            if k in out.columns:
                out[k] = pd.to_datetime(out[k])
    return out


def decompress_string(sz: str, verbose=False):
    """
    Decompress strings encoded in `patdb_tbox.psql.psql.compress_string`

    Inputs

    - sz:str, utf8 character strings (see `patdb_tbox.psql.psql.compress_string`)
    - verbose: bool
    """

    if verbose:
        pidprint("Get base64 from zstring...")
    out = base64.b64decode(sz)

    if verbose:
        pidprint("Get decompressed bytes...")
    out = zlib.decompress(out)

    if verbose:
        pidprint("Decode string")
    return out.decode("utf8")


def compress_string(s: str):
    """Compress a string of utf8 characters.

    Procedure:

    - encode as a utf8 bytes
    - compress bytes with `zlib.compress`
    - encode bytes as base64 with `base64.b64encode`
    - decode bytes as utf8 characters.
    """
    return base64.b64encode(zlib.compress(s.encode("utf8"))).decode("utf8")


def read_compressed_datafile(fname: str, verbose=False, nrows: int = None) -> pd.DataFrame:
    """
    Read compressed datafile.
    The data file might be compressed with 'gzip' or with an old silly method defined in `patdb_tbox.psql.psql.compress_chunk`.
    """
    if verbose:
        pidprint("Start reading...")
    try:
        pidprint("New version", flag="info")
        F = pd.read_csv(fname, sep=";", compression='gzip')
    except:
        pidprint("Old version", flag="info")
        
        with open(fname, "r") as fp:
            F_str = fp.read()
        if verbose:
            pidprint("Start decompressing...")
        F = decompress_chunk(F_str, verbose=verbose, nrows=nrows)
        if verbose:
            pidprint("Finished decompressing.")
        
    F = parse_datafile(F)
    pidprint("Finished reading...")
    return F


def parse_datafile(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Parse the potential time columns of a dataframe.
    The potential columns are `timestamp` and `context__tl`
    """
    if verbose:
        pidprint("Parse time columns...")
    for k in ["timestamp", "context__tl"]:
        if k in df.columns:
            df[k] = pd.to_datetime(df[k])
    return df


def write_compressed_datafile(F: pd.DataFrame, fname: str):
    """
    Write `pd.DataFrame` to csv file with 'gzip' compress algorithm and ';' separators.
    """
    F.to_csv(fname, sep=";", compression="gzip")

### Copied from Caorlins utils file ###
def pidprint(*arg, flag="status"):
    """
    Behaves like builtin `print` function, but append runtime info before writing to stderr.
    The runtime info is:

    - pid
    - datetime.now()
    - flag, specified as a keyword
    """
    print("[{}] [{}] [{}]".format(os.getpid(),datetime.now(), flag)," ".join(map(str, arg)), file=sys.stderr)
    return
### Copied from Carolins utils file ###
"""
### Henriks playground ###
def main():
    # file_path = "Z:/Groups/Eric Herlenius/henrik/51061fe4eb0cb9da61ebd13d39879eefb5bfe33f103d3a80c7123deb5e1cd8c9.dat"
    file_name = "51061fe4eb0cb9da61ebd13d39879eefb5bfe33f103d3a80c7123deb5e1cd8c9.dat"
    print("Reading file:", file_name)
    df = read_compressed_datafile(file_name)
    print("Columns:\n", len(df.columns), df.columns)
    print("Data:\n", df.head)

main()
"""