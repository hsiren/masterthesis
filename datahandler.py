# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:13:57 2022

@author: hensir
"""
from utils import *
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

FEAT_PATHS = ["51061fe4eb0cb9da61ebd13d39879eefb5bfe33f103d3a80c7123deb5e1cd8c9.dat"]
RAW_PATHS = ["fcf69cb7cdb43764ae519f06fb1de1ac814c4525a2875d947a3494faccdd3734.dat"]

DATA_PATHS = FEAT_PATHS
SEQUENCE_LENGTH = 10

#WEIRD = Unnamed: 0"context__tl"] group__uid context__tkevt_tl demos__birthdate context__time_to_event_h
LABEL = "target__y"
FEAT_LIST_SHORT = ["feats__btb_mean", "feats__rf_mean", "feats__spo2_mean"]
FEAT_LIST = ["feats__btb_mean", 
             "feats__rf_mean", 
             "feats__spo2_mean", 
             "feats__btb_std",
             "feats__rf_std",
             "feats__spo2_std",
             "feats__btb_max",
             "feats__rf_max",
             "feats__spo2_max",
             "feats__btb_min",
             "feats__rf_min",
             "feats__spo2_min",
             "feats__btb_skew",
             "feats__rf_skew",
             "feats__spo2_skew",
             "feats__btb_kurtosis",
             "feats__rf_kurtosis",
             "feats__spo2_kurtosis",
             "feats__btb_sampAs",
             "feats__btb_sampEn",
             "feats__cirk_vikt",
             "feats__bw",
             "feats__sex",
             "feats__pnage_days"]

def readData(file_path):
    
    print("Reading file:", file_path)
    df = read_compressed_datafile(file_path)
    print("Features:\n", len(df.columns))
    for col in df.columns:
        print(col)
    print("Data:\n", df.head)

    print("Plotting:")
    #df[["feats__btb_mean","feats__rf_mean","feats__rf_std","target__y",]].iloc[1:100].plot.line(subplots=True)
    print("Plotting done.")
    print(df["target__y"].unique())
    return df

# Do preprocessing:
# TODO: Normmalize data.
def preprocess(data, plot=False):
    print("min:", data["feats__btb_mean"].min(), data["feats__rf_mean"].min())
    # Replace missing values with NaN
    rep_val = math.nan
    data = data.replace(data["feats__btb_mean"].min(), rep_val)#float("nan"))
    data = data.replace(data["feats__rf_mean"].min(), rep_val)#float("nan"))
    
    if plot:
        print("feats__btb_mean:", data["feats__btb_mean"].min())
        print("feats__rf_mean:", data["feats__rf_mean"].min())
        print("feats__spo2_mean:", data["feats__spo2_mean"].min())
        print("feats__btb_std:", data["feats__btb_std"].min())
        print("feats__rf_std:", data["feats__rf_std"].min())
        print("feats__spo2_std:", data["feats__spo2_std"].min())
        print("feats__btb_max:", data["feats__btb_max"].min())
        print("feats__rf_max:", data["feats__rf_max"].min())
        print("feats__spo2_max:", data["feats__spo2_max"].min())
        print("feats__btb_min:", data["feats__btb_min"].min())
        print("feats__rf_min:", data["feats__rf_min"].min())
        print("feats__spo2_min:", data["feats__spo2_min"].min())
        print("feats__btb_skew:", data["feats__btb_skew"].min())
        print("feats__rf_skew:", data["feats__rf_skew"].min())
        print("feats__spo2_skew:", data["feats__spo2_skew"].min())
        print("feats__btb_kurtosis:", data["feats__btb_kurtosis"].min())
        print("feats__rf_kurtosis:", data["feats__rf_kurtosis"].min())
        print("feats__spo2_kurtosis:", data["feats__spo2_kurtosis"].min())
        print("feats__btb_sampAs:", data["feats__btb_sampAs"].min())
        print("feats__btb_sampEn:", data["feats__btb_sampEn"].min())
        print("feats__cirk_vikt:", data["feats__cirk_vikt"].min())
        print("feats__bw:", data["feats__bw"].min(), data["feats__bw"].unique())
        print("feats__sex:", data["feats__sex"].min())
        print("feats__pnage_days:", data["feats__pnage_days"].min())

    return data

# Turn data into sequence length tuples with feats and labels.
def sequenceTuples(data, n_sequences=None, add_rest=False):
    sequence_tuples = []
    n = len(data)
    n_modulo = n % SEQUENCE_LENGTH
    n_full_sequences = int((n-n_modulo)/SEQUENCE_LENGTH)
    
    print(n, n_modulo)
    print(n_full_sequences)
    for i in tqdm(range(n_full_sequences)):
        feats = data[FEAT_LIST_SHORT][i:i+SEQUENCE_LENGTH]
        labels = data[LABEL][i:i+SEQUENCE_LENGTH]
        #print("FEATS:", feats.shape)
        #print("LABELS:", labels.shape)
        sequence_tuples.append( (feats, labels) )
        
        if n_sequences is not None:
            if i > n_sequences:
                break
        #break
    
    # TODO: Fix
    if add_rest:
        if n_modulo > 0:
            print(data[FEAT_LIST_SHORT].shape)
            feats = data[FEAT_LIST_SHORT][n_full_sequences*SEQUENCE_LENGTH:]
            labels = data[FEAT_LIST_SHORT][n_full_sequences*SEQUENCE_LENGTH:]
            sequence_tuples.append( (feats, labels) )
            print("FEATS_rest:", feats.shape)
            print("LABELS_rest:", labels.shape)
    
    #for i in range()
    return sequence_tuples

# data = readData(DATA_PATHS[0])
# data = preprocess(data)
# data_tuples = sequenceTuples(data)