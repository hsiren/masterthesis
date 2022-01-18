from utils import *
import pandas as pd
import matplotlib.pyplot as plt
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 14:46:50 2021

@author: hensir
"""

### Henriks playground ###
def main():
    """
    Note: 
        - First index seems to be brutally low for all first indexes.
    """
    # file_path = "Z:/Groups/Eric Herlenius/henrik/51061fe4eb0cb9da61ebd13d39879eefb5bfe33f103d3a80c7123deb5e1cd8c9.dat"

    # Data: extracted features
    file_name = "51061fe4eb0cb9da61ebd13d39879eefb5bfe33f103d3a80c7123deb5e1cd8c9.dat"
    print("Reading file:", file_name)
    df = read_compressed_datafile(file_name)
    print("Columns:\n", len(df.columns), df.columns)
    print("Data:\n", df.head)
    print("feats__rf_mean:\n", df["feats__rf_mean"])
    print("Plotting:")
    df[["feats__btb_mean","feats__rf_mean","feats__rf_std","target__y",]].iloc[1:100].plot.line(subplots=True)
    df[["target__y",]].iloc[1:10000].plot.line(subplots=True)
    print("Plotting done.")
    print(df["target__y"].unique())

    
    """
    # Data: raw
    file_name_raw = "fcf69cb7cdb43764ae519f06fb1de1ac814c4525a2875d947a3494faccdd3734.dat"
    print("Reading raw_data:")
    df_raw = read_compressed_datafile(file_name_raw, verbose=True, nrows=200)
    print("Reading done.")
    print("Columns_raw:\n", len(df_raw.columns), df_raw.columns)
    print("Data_raw:\n", df_raw.head)
    df_raw[["feats__d0","feats__d1","target__y"]].iloc[1:100].plot.line(subplots=True)

    #df[["feats__btb_mean","feats__rf_mean","feats__rf_std"]].iloc[1:10].plot.line(subplots=True)
    """
main()