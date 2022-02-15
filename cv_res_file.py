#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import pickle as pkl
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from functools import partial
import gc
#%matplotlib notebook


def read_file(fname, figs=True):
    params, res = pd.read_pickle(fname)
    if not figs:
        del res["figs"]
    gc.collect()
    gc.collect()
    return pd.DataFrame([{**params["fit"], **params["data"], **params["mdl"], **res}]).to_csv(sys.stdout, sep=";")


def see_fixed(df):
    for k in df.columns:
        if not any([s == k for s in ["end_Q", "fit_time"]]):
            print(k, ":", df[k].unique())


def read_results(exp_dir, root_dir="..", verbose=False, nlim=None, figs=True):
    indir = os.path.join(root_dir, "res", exp_dir)
    all_fnames = glob(os.path.join(indir, "*.pkl"))
    print(len(all_fnames), "files to read,", "limit to:", nlim)

    read_fun = partial(read_file, figs=figs)
    all_data=[]
    #df=pd.DataFrame()
    plt.ioff()

    for i,fname in enumerate(all_fnames[:nlim]):
        if i==0:
            pd.DataFrame([read_fun(fname)]).to_csv(sys.stdout, sep=";")
        else:
            pd.DataFrame([read_fun(fname)]).to_csv(sys.stdout, sep=";",header=None)
        plt.close()
        #df=df.append()
        #gc.collect()
        #gc.collect()
        #if (df.shape[0] % 10)==0:
        #    print(df.shape[0],"/",nlim)

    if verbose:
        print("The files are read.")

    df = pd.DataFrame(all_data)
    if verbose:
        print(df.memory_usage().sum() / 1024 / 1024, "MB")
        see_fixed(df)
    return df


def main(args):
    read_file(args.i,figs=False)
    #df = read_results("conv2d_2",root_dir="..",verbose=False,figs=False,nlim=None)
    #df.to_csv("conv2d_2__all_results.csv.gz",sep=";")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str,required=True, help="Input pkl")


if __name__ =="__main__":
    args=parser.parse_args()
    main(args)