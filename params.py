# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:56:05 2022

@author: hensir
"""

# --- Preprocessing --- #
# Interpolation threshold:
#THRESHOLD = 60
#SEQUENCE_LENGTH = 10
# --- Preprocessing --- #

# --- Other static params --- #
#LEARNING_RATE = 0.001
# --- Other static params --- #

# --- RNN --- #
# Hyperparameters:
#EPOCHS = 400
#HIDDEN_DIM = 24
#LAYER_DIM = 1
#LEARNING_RATE = 0.001
# Loss weight:
#MAJORITY_CLASS_WEIGHT = 0.6#
#MINORITY_CLASS_WEIGHT = 1.4#
#SINGLE_OUTPUT = True
#N_SEQUENCES = 5000
# --- RNN --- #

# Paths to data:
FEAT_PATHS = ["51061fe4eb0cb9da61ebd13d39879eefb5bfe33f103d3a80c7123deb5e1cd8c9.dat"]
RAW_PATHS = ["fcf69cb7cdb43764ae519f06fb1de1ac814c4525a2875d947a3494faccdd3734.dat"]

# Imported path to data:
DATA_PATHS = FEAT_PATHS
#DATA_PATHS = RAW_PATHS

# Name of saved preprocessed dataframe:
DATA_FILENAME = "Preprocessed_feats"

# Config directory name:
CFG_DIR = "configs\cfgs.json"

#WEIRD = Unnamed: 0"context__tl"] group__uid context__tkevt_tl demos__birthdate context__time_to_event_h
# Features and labels:
LABEL = "target__y"
FEAT_LIST_SHORT = ["feats__btb_mean", "feats__rf_mean", "feats__spo2_mean"]
FEAT_LIST_ALL = ["feats__btb_mean", "feats__rf_mean", "feats__spo2_mean", "feats__btb_std", "feats__rf_std", "feats__spo2_std", "feats__btb_max", "feats__rf_max", "feats__spo2_max", "feats__btb_min", "feats__rf_min", "feats__spo2_min", "feats__btb_skew", "feats__rf_skew", "feats__spo2_skew", "feats__btb_kurtosis", "feats__rf_kurtosis", "feats__spo2_kurtosis", "feats__btb_sampAs", "feats__btb_sampEn", "feats__cirk_vikt", "feats__bw", "feats__sex", "feats__pnage_days"]

# Imported feature list:
#FEAT_LIST = FEAT_LIST_SHORT
FEAT_LIST = FEAT_LIST_ALL

"""
####
1 feats__btb_mean,
2 feats__btb_std, 
3 feats__btb_skew,
4 feats__btb_kurtosis, 
5 feats__btb_sampEn, 
6 feats__btb_sampAs,
7 feats__btb_min, 
8 feats__btb_max,  

9 feats__rf_std, 
10 feats__rf_kurtosis, 
11 feats__rf_min, 
12 feats__rf_skew, 
13 feats__rf_max, 
14 feats__rf_mean, 

15 feats__spo2_max, 
16 feats__spo2_mean, 
17 feats__spo2_skew,  
18 feats__spo2_std, 
19 feats__spo2_kurtosis, 
20 feats__spo2_min,

feats__cirk_vikt, 
feats__pnage_days, 
feats__bw, 
feats__sex, 


####
feats__btb_mean
feats__rf_mean
feats__spo2_mean
feats__btb_std
feats__rf_std
feats__spo2_std
feats__btb_max
feats__rf_max
feats__spo2_max
feats__btb_min
feats__rf_min
feats__spo2_min
feats__btb_skew
feats__rf_skew
feats__spo2_skew
feats__btb_kurtosis
feats__rf_kurtosis
feats__spo2_kurtosis
feats__btb_sampAs
feats__btb_sampEn
feats__cirk_vikt
feats__bw
feats__sex
feats__pnage_days

target__y
group__uid
context__tkevt_tl
demos__birthdate
context__time_to_event_h
"""