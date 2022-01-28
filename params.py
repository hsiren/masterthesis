# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:56:05 2022

@author: hensir
"""


"""
TODO:
General:
    [] Dynamic sequence splitting!
    [] Modularize
datahandler.py:
    [] Handle raw data:
        [] Save functionality for raw data for faster processing
    [] Make compatible with GPU
    [-] Handle data split based on n-sequences (not n_patients)
RNN.py:
    [X] Adjust model for batched data:
        [x] Make compatible with GPU
    [x] Fix loss function data type
    [] Backprop per sequence (not per batch)
"""

# Paths to data:
FEAT_PATHS = ["51061fe4eb0cb9da61ebd13d39879eefb5bfe33f103d3a80c7123deb5e1cd8c9.dat"]
RAW_PATHS = ["fcf69cb7cdb43764ae519f06fb1de1ac814c4525a2875d947a3494faccdd3734.dat"]

# Imported path to data:
DATA_PATHS = FEAT_PATHS
#DATA_PATHS = RAW_PATHS

# Hyperparameters:
SEQUENCE_LENGTH = 10
EPOCHS = 100
HIDDEN_DIM = 500
LAYER_DIM = 1
LEARNING_RATE = 0.0001
# Loss weight:
MAJORITY_CLASS_WEIGHT = 1.0
MINORITY_CLASS_WEIGHT = 1.0
SINGLE_OUTPUT = False
#N_SEQUENCES = 5000


#WEIRD = Unnamed: 0"context__tl"] group__uid context__tkevt_tl demos__birthdate context__time_to_event_h
# Features and labels:
LABEL = "target__y"
FEAT_LIST_SHORT = ["feats__btb_mean", "feats__rf_mean", "feats__spo2_mean"]
FEAT_LIST_ALL = ["feats__btb_mean", 
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

# Imported feature list:
#FEAT_LIST = FEAT_LIST_SHORT
FEAT_LIST = FEAT_LIST_ALL