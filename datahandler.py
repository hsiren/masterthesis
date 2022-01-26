# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:13:57 2022

@author: hensir
"""
from utils import *
import  numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
import torch
from torch import nn

from params import DATA_PATHS, FEAT_LIST, SEQUENCE_LENGTH, LABEL, SEQUENCE_LENGTH


def readData(file_path, raw=False):
    
    if raw:
        print("Reading file:", file_path)
        df = read_compressed_datafile(file_path)
        print("Features:\n", len(df.columns))
        print("Data:", df.shape)
        
        for i,c in enumerate(df.columns):
            if "feats__d" not in c:
                print("i:", i, "c: ["+str(c)+"]")
        
        # 9910 totalt
        
        print(df.head)
        print("Extracting X & Y...")        
        
        # Each row of F contains a flattened 3d time series + context info
        X = df[[s for s in df.columns if "feats__" in s]].values
        # X = df[cols].values #Do not use!
        # X = df[[s for s in df.columns if "feats__" in s]].values
        #print("X:", X.shape)
        #x = X[0].reshape((-1, 5))
        print("X:", X.shape)

        Y = df[["target__y"]].values.reshape(-1, 1)
        print("Y:", Y.shape)

        #Z = df.values.reshape(-1,3)
        print("Reshaping...")
        x = X[0].reshape(-1, 3)
        print("X:", X.shape)
        y = Y[0]
        print("Y:", Y.shape)

        # x is an example of a 2Hz, 3d time series of length 60 min 
        # ndarray of size (n_samples, 3) 
    
        print("Stuff...")
        n_samples, d = x.shape
        sample_per_sec = 2
        time_sec = np.arange(n_samples) / sample_per_sec
        signames = ["btb", "rf", "spo2"]

        print("Plot...")
        plt.figure()
        plt.plot(time_sec, x.reshape(-1, 3))
        plt.legend(signames)
        plt.title("Target:{}".format(y))
        plt.savefig("Example signals.pdf")
        plt.close()
        raise Exception("Raw-stop")
    else:
        print("Reading file:", file_path)
        df = read_compressed_datafile(file_path)
        print("Features:\n", len(df.columns))
        for col in df.columns:
            print(col)
        print("Data:\n", df.head)
        print(df.shape)
        # (134668,31)
        print("Plotting:")
        #df[["feats__btb_mean","feats__rf_mean","feats__rf_std","target__y",]].iloc[1:100].plot.line(subplots=True)
        print("Plotting done.")
        print(df["target__y"].unique())
        print("Group__uid:\n", len(df["group__uid"].unique()), df["group__uid"].unique())
    return df

# Do preprocessing:
# TODO: Normmalize data.
def preprocess(data, print_minmax=False):
    print("min:", data["feats__btb_mean"].min(), data["feats__rf_mean"].min())
    
    # Replace missing values with NaN
    rep_val = math.nan
    data = data.replace(data["feats__btb_mean"].min(), rep_val)#float("nan"))
    data = data.replace(data["feats__rf_mean"].min(), rep_val)#float("nan"))
    #print("labels:", labels.shape)
    print("data:", data.shape, data.head())
    # Remove first value
    #data = data.drop(0, axis=0)
    
    # print min max vals in data
    if print_minmax:
        print("feats__btb_mean:", data["feats__btb_mean"].min(), data["feats__btb_mean"].max())
        print("feats__rf_mean:", data["feats__rf_mean"].min(), data["feats__rf_mean"].max())
        print("feats__spo2_mean:", data["feats__spo2_mean"].min(), data["feats__spo2_mean"].max())
        print("feats__btb_std:", data["feats__btb_std"].min(), data["feats__btb_std"].max())
        print("feats__rf_std:", data["feats__rf_std"].min(), data["feats__rf_std"].max())
        print("feats__spo2_std:", data["feats__spo2_std"].min(), data["feats__spo2_std"].max())
        print("feats__btb_max:", data["feats__btb_max"].min(), data["feats__btb_max"].max())
        print("feats__rf_max:", data["feats__rf_max"].min(), data["feats__rf_max"].max())
        print("feats__spo2_max:", data["feats__spo2_max"].min(), data["feats__spo2_max"].max())
        print("feats__btb_min:", data["feats__btb_min"].min(), data["feats__btb_min"].max())
        print("feats__rf_min:", data["feats__rf_min"].min(), data["feats__rf_min"].max())
        print("feats__spo2_min:", data["feats__spo2_min"].min(), data["feats__spo2_min"].max())
        print("feats__btb_skew:", data["feats__btb_skew"].min(), data["feats__btb_skew"].max())
        print("feats__rf_skew:", data["feats__rf_skew"].min(), data["feats__rf_skew"].max())
        print("feats__spo2_skew:", data["feats__spo2_skew"].min(), data["feats__spo2_skew"].max())
        print("feats__btb_kurtosis:", data["feats__btb_kurtosis"].min(), data["feats__btb_kurtosis"].max())
        print("feats__rf_kurtosis:", data["feats__rf_kurtosis"].min(), data["feats__rf_kurtosis"].max())
        print("feats__spo2_kurtosis:", data["feats__spo2_kurtosis"].min(), data["feats__spo2_kurtosis"].max())
        print("feats__btb_sampAs:", data["feats__btb_sampAs"].min(), data["feats__btb_sampAs"].max())
        print("feats__btb_sampEn:", data["feats__btb_sampEn"].min(), data["feats__btb_sampEn"].max())
        print("feats__cirk_vikt:", data["feats__cirk_vikt"].min(), data["feats__cirk_vikt"].max())
        print("feats__bw:", data["feats__bw"].min(), data["feats__bw"].max())
        print("feats__sex:", data["feats__sex"].min(), data["feats__sex"].max())
        print("feats__pnage_days:", data["feats__pnage_days"].min(), data["feats__pnage_days"].max())
        #print("BW (unique):",  data["feats__bw"].unique())

    # --- Normalize between -1 and 1 --- #
    # Get features from data
    prescaled_data = data[FEAT_LIST]
    
    # Separate labels (they are not scaled)
    labels = data[LABEL]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(prescaled_data)
    scaled_data = scaler.transform(prescaled_data)
    scaled_data = pd.DataFrame(scaled_data, columns=FEAT_LIST)
    scaled_data[LABEL] = labels
    #data = scaled_data
    # --- Normalize between -1 and 1 --- #

    # Add group id to scaled data:
    scaled_data["group__uid"] = data["group__uid"]
    data = scaled_data
    
    #data = scaled_data.insert(0, "group__uid", data["group__uid"])
    #print(data.shape, data.head)

    # Remove first value (has nothing)
    data = data.drop(0, axis=0)
    
    #print(data.shape, data.head)
    return data

def devicePrint(device):
    mode = None
    if device == "cuda":
        mode = "GPU"
    else:
        mode = "CPU"
    print("===== {Running on} =====")
    print("\t\t {"+device+"} ")
    print("===== {Running on} =====")
    
def quickPrint(df, name="quickPrint"):
    print("---", name, "---")
    for col in df.columns:
        print(col)
    print("Data:\n", df.head)
    print(df.shape)
    print("---", name, "---")

# --- Depricated --- #
# Turn data into sequence length tuples with feats and labels.
def sequenceTuples(data, n_sequences=None, add_rest=False):
    sequence_tuples = []
    n = len(data)
    n_modulo = n % SEQUENCE_LENGTH
    n_full_sequences = int((n-n_modulo)/SEQUENCE_LENGTH)
    
    print(n, n_modulo)
    print(n_full_sequences)
    for i in tqdm(range(n_full_sequences)):
        feats = data[FEAT_LIST][i:i+SEQUENCE_LENGTH]
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
            print(data[FEAT_LIST].shape)
            feats = data[FEAT_LIST][n_full_sequences*SEQUENCE_LENGTH:]
            labels = data[FEAT_LIST][n_full_sequences*SEQUENCE_LENGTH:]
            sequence_tuples.append( (feats, labels) )
            print("FEATS_rest:", feats.shape)
            print("LABELS_rest:", labels.shape)
    
    #for i in range()
    return sequence_tuples
# --- Depricated --- #

def batchify(data):
    print("Group__uid:\n", len(data["group__uid"].unique()), data["group__uid"].unique())
    patients = data["group__uid"].unique()
    patient_batches = []
    for patient in patients:
        #query_string = 'group__uid=='+patient
        #print("{"+query_string+"}")
        patient_data=data.loc[data['group__uid'] == patient]
        print("Points/patient:", patient_data.shape[0])
        #print(patient_data.columns)
        #df = data.query(query_string)
        #print(patient_data.head)
        patient_batches.append(patient_data)
        #raise Exception("LOL")
    print("patients:", len(patient_batches))
    return patient_batches

def dsequencify(batched_data):
    sequenced_batches = []
    n_batches = len(batched_data)
    
    for batch in tqdm(batched_data):
        n = len(batch)
        n_modulo = n % SEQUENCE_LENGTH
        batch_sequence_tuples = []
        
        # If len(sequence) < SEQUENCE_LENGTH
        if n < SEQUENCE_LENGTH:
            feats = batch[FEAT_LIST]
            labels = batch[LABEL]
            batch_sequence_tuples.append( (feats,labels) )
            
        # If len(sequence) >= SEQUENCE_LENGTH
        elif n >= SEQUENCE_LENGTH:
            n_modulo = n % SEQUENCE_LENGTH
            n_full_sequences = int((n-n_modulo)/SEQUENCE_LENGTH)
            
            for i in range(n_full_sequences):
                start = i*SEQUENCE_LENGTH
                end = i*SEQUENCE_LENGTH + SEQUENCE_LENGTH
                feats = batch[FEAT_LIST][start:end]
                labels = batch[LABEL][start:end]
                batch_sequence_tuples.append( (feats, labels) )

            if n-n_modulo != 0:
                feats = batch[FEAT_LIST][(i+1)*SEQUENCE_LENGTH:]
                labels = batch[LABEL][(i+1)*SEQUENCE_LENGTH:]
                batch_sequence_tuples.append( (feats, labels) )
                
        sequenced_batches.append(batch_sequence_tuples)
    
    n_sequence_batches = len(sequenced_batches)
    
    if n_batches != n_sequence_batches:
        print("n_batches:", n_batches, "n_sequence_batches:", n_sequence_batches)
        raise Exception("Error in sequencing")
    return sequenced_batches

def sequencify(batched_data):
    sequenced_batches = []
    n_batches = len(batched_data)
    print("n_batches:", n_batches)
    #switch = True
    #temp = None
    it_NaN_batches = 0
    for batch in tqdm(batched_data):
        n = len(batch)
        #n_modulo = n % SEQUENCE_LENGTH
        #n_full_sequences = int((n-n_modulo)/SEQUENCE_LENGTH)
        batch_sequence_tuples = []
        batch_sequence_feats = []
        batch_sequence_labels = []
        
        #print("n:", n)
        #print("batch:", batch.shape)
        #batch = 
        if batch.isnull().all().all():
            print("All NaN", )
        batch[FEAT_LIST] = batch[FEAT_LIST].interpolate(method='linear', 
                          axis=0, # 0: (column)
                          limit=None, 
                          inplace=False, 
                          limit_direction="both", 
                          limit_area=None, 
                          downcast=None)
        """
        print("batch:", batch.shape)
        print(batch[FEAT_LIST])
        print(batch[FEAT_LIST].isnull().values.any())
        print(batch.isnull().values.any())
        raise Exception("Pause")
        """
        if batch.isnull().values.any():
            it_NaN_batches += 1
            print("Full NaN-batch:", it_NaN_batches)
            #print(batch.shape)
            #print(batch[FEAT_LIST])
            continue

        # Torch tensor shape: (N, L, H_in)
        #N = n; L = SEQUENCE_LENGTH; H_in = len(FEAT_LIST)
        if n < SEQUENCE_LENGTH:
            feats = batch[FEAT_LIST]
            labels = batch[LABEL]
            #print("---------------")
            #print("Feats1:", feats)
            #print("Feats2:", feats.values)
            feats = torch.Tensor(feats.values)
            labels = torch.Tensor(labels.values)
            #labels = torch.reshape(labels, (1,len(labels)))
            labels = torch.reshape(labels, (len(labels),1))

            #print("Type:", feats)
            #print("---------------")
            # TODO: do labels
            # TODO: Check for all NaN!
            #batch_sequence_tuples.append( (feats,labels) )
            
            batch_sequence_feats.append(feats)
            batch_sequence_labels.append(labels)

        
        elif n >= SEQUENCE_LENGTH:
            n_modulo = n % SEQUENCE_LENGTH
            n_full_sequences = int((n-n_modulo)/SEQUENCE_LENGTH)
            
            for i in range(n_full_sequences):
                start = i*SEQUENCE_LENGTH
                end = i*SEQUENCE_LENGTH + SEQUENCE_LENGTH
                
                feats = batch[FEAT_LIST][start:end]
                labels = batch[LABEL][start:end]
                
                #print("Feats1:", type(feats), feats.shape, feats)
                #print("Feats2:", feats.values)
                feats = torch.Tensor(feats.values)
                labels = torch.Tensor(labels.values)
                #labels = torch.reshape(labels, (1,len(labels)))
                labels = torch.reshape(labels, (len(labels),1))

                #print("Type:", type(feats), feats.shape, feats)
                #raise Exception("PAUSE")
                #batch_sequence_tuples.append( (feats, labels) )
                
                batch_sequence_feats.append(feats)
                batch_sequence_labels.append(labels)



            if n-n_modulo != 0:
                feats = batch[FEAT_LIST][(i+1)*SEQUENCE_LENGTH:]
                labels = batch[LABEL][(i+1)*SEQUENCE_LENGTH:]
                feats = torch.Tensor(feats.values)
                labels = torch.Tensor(labels.values)
                #labels = torch.reshape(labels, (1,len(labels)))
                labels = torch.reshape(labels, (len(labels),1))

                #print("---------------")
                #print("Feats1:", feats)
                #print("Feats2:", feats.values)
                #print("Type:", feats)
                #print("---------------")
                # TODO: do labels
                #batch_sequence_tuples.append( (feats, labels) )
                
                #print("f1:", feats.shape)
                #print("l1:", labels.shape, labels)
                # Pad label
                #left_padding = 0; right_padding = 4
                #label_padding = nn.ConstantPad1d((left_padding, right_padding), 0)
                #print("l15:", label_padding)
                #labels = label_padding(labels.T)
                #print("l2:", labels.shape, labels)

                batch_sequence_feats.append(feats)
                batch_sequence_labels.append(labels)
                
                #if switch:
                    #print("b1:", feats.shape)
                    #print("b1:", feats)
                    #temp = feats
                    #switch = False
                
        #batch_feats = batch[FEAT_LIST]
        # T: longest sequence
        # B: batch size
        # *: Any number of dimensions
        #print("TEST", batch_sequence_labels[0].shape)
        #for i in batch_sequence_labels:
            #print("labels:", i.shape)
        #label1 = torch.reshape(batch_sequence_labels[0], (1,len(batch_sequence_labels[0])))
        #print("label1:", label1.shape, label1)
        
        padded_batch_labels = nn.utils.rnn.pad_sequence(batch_sequence_labels, batch_first=True, padding_value=0.0)
        #print(padded_batch_labels.shape)
        
        #raise Exception("LLL")
        #raise Exception("Pause1")
        # Pad sequences:
        padded_batch_feats = nn.utils.rnn.pad_sequence(batch_sequence_feats, batch_first=True, padding_value=0.0)
        #print("feats:", padded_batch_feats.size(), "labels:", padded_batch_labels.size())
        #print("fff:", padded_batch_feats.shape[0])
        #for i in range(padded_batch_feats.shape[0]):
            #print("feats:", padded_batch_feats, "labels:", padded_batch_labels)
        #raise Exception("Pause")
            
        sequenced_batches.append( (padded_batch_feats, padded_batch_labels) )
        
    #print("TEST1:")
    #print(temp.shape)
    #print(temp)  
    #print("TEST2:")
    #print(sequenced_batches[-1][0].shape)
    #print(sequenced_batches[0][-1])
    #ind = 3
    #print("Feats:", sequenced_batches[ind][0].shape)
    #print("Labels:", sequenced_batches[ind][1].shape)
    #print("Feats:", sequenced_batches[ind][0])
    #print("Labels:", sequenced_batches[ind][1])
    
    #print("sequenced_batches:", sequenced_batches[-2])
    #print(len(sequenced_batches), n_batches, len(sequenced_batches)+it_NaN_batches == n_batches)
    # raise Exception("PAUSE FINAL")
    return sequenced_batches

def splitData(sequenced_batches, test_size=0.3, shuffle=False):
    n_batches = len(sequenced_batches)
    #print("N_batches:", n_batches)
    
    n_test = int(round(test_size * n_batches))
    n_train = n_batches - n_test
    if shuffle:
        random.shuffle(sequenced_batches)
    
    """
    # --- Check sentence lengths --- #
    n_sequences = 0
    ind = 0
    sorting_dict = {}
    batch_sizes = []
    for batch in sequenced_batches:
        #print("batch:", batch[0].shape)
        n_sequence = batch[0].shape[0]
        batch_sizes.append(n_sequence)
        sorting_dict[str(ind)] = batch
        n_sequences += n_sequence
        ind += 1
        
    print("n_sequences:", n_sequences, "test:", int(round(n_sequences*test_size)), "train:", n_sequences-int(round(n_sequences*test_size)))
    test = int(round(n_sequences*test_size))
    train = n_sequences-int(round(n_sequences*test_size))
    
    # Match nr patients with nr sequences
    train = []
    test = []
    button = True
    for i in range(len(batch_sizes)):
        max_val = max(batch_sizes)
        batch_sizes.remove(max_val)
        if button:
            train.append(sorting_dict[str(i)])
            button = False
        else:
            test.append(sorting_dict[str(i)])
            button = True
    
    print("train:", len(train))
    print("test:", len(test))
    n_train_sequences = 0
    for train_batch in train:
        #print("batch:", batch[0].shape)
        n_train_sequence = train_batch[0].shape[0]
        n_train_sequences += n_train_sequence
    print("n_train_sequences:", n_train_sequences)

    n_test_sequences = 0
    for test_batch in test:
        #print("batch:", batch[0].shape)
        n_test_sequence = test_batch[0].shape[0]
        n_test_sequences += n_test_sequence
    print("n_test_sequences:", n_test_sequences)
    raise Exception("PAUSE")
    """
    train = sequenced_batches[:n_train]
    test = sequenced_batches[n_train:]
    
    #print("Train:", len(train), "Test:", len(test), "(train+test)", len(train)+len(test), n_batches)
    return train, test

"""
data = readData(DATA_PATHS[0], raw=False)
data = preprocess(data, print_minmax=True)
quickPrint(data)
batched_data = batchify(data)
sequenced_batches = sequencify(batched_data)
train, test, splitData(sequenced_batches, test_size=0.3, shuffle=False)
"""
# sequenced_batches = sequencify(batched_data)


# --- Depricated --- #
#data_tuples = sequenceTuples(data, 1000)
# --- Depricated --- #
