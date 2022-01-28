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
    print("{"+device+"} ")
    print("===== {Running on} =====")
    
def quickPrint(df, name="quickPrint"):
    print("---", name, "---")
    for col in df.columns:
        print(col)
    print("Data:\n", df.head)
    print(df.shape)
    
    print("group__uid:\n", df["target__y"].unique())
    print("group__uid:\n", df["target__y"].value_counts())

    
    """
    patients = df["group__uid"].unique()
    positive_labels_in_batch = 0
    sepsis_patients = []
    for patient in patients:
        patient_data=df.loc[df['group__uid'] == patient]
        #print(patient_data.shape)
        #print("----- Patient -----")
        #print(patient)
        #print("Points/patient:", patient_data.shape[0])
        #print("Labels:", patient_data["target__y"].unique())
        vc = patient_data["target__y"].value_counts()
        #print("Value counts:\n", vc, vc.shape[0])
        if vc.shape[0] == 2:
            positive_labels_in_batch += 1
            sepsis_patients.append(patient)
            print("Points/patient:", patient_data.shape[0])
        #print("----- Patient -----")
    print("positive_labels_in_batch:", positive_labels_in_batch)
    #"""
    print("---", name, "---")

def batchify(data):
    print("Group__uid:\n", len(data["group__uid"].unique()), data["group__uid"].unique())
    patients = data["group__uid"].unique()
    patient_batches = {}
    sepsis_patients = []
    switch = True
    total_points = 0
    for patient in patients:
        patient_data=data.loc[data['group__uid'] == patient]
        vc = patient_data["target__y"].value_counts()
        if vc.shape[0] == 2:
            sepsis_patients.append(patient)
            print("---- " + patient + " ----")
            print("Points/patient (sepsis):", patient_data.shape[0], "0:", vc[0], "1:", vc[1])
            x = np.arange(patient_data["target__y"].shape[0])
            plt.plot(x, patient_data["target__y"])
            plt.title(patient)
            plt.show()
            print("---------------------------------------------")

        else:
            print("Points/patient:", patient_data.shape[0])
        total_points += patient_data.shape[0]

        patient_batches[patient] = patient_data
    print("patients:", len(patient_batches))
    print("Batchify total_points:", total_points)
    return patient_batches, sepsis_patients

def sequencify(batched_data):
    sequenced_batches = {}
    n_batches = len(batched_data)
    print("n_batches:", n_batches, len(batched_data.keys()))
    it_NaN_batches = 0
    points_in_NaN = 0
    if1 = 0; if2 = 0
    total = 0
    for patient in tqdm(batched_data.keys()):
        n = len(batched_data[patient])
        batch_sequence_tuples = []
        batch_sequence_feats = []
        batch_sequence_labels = []
        
        batched_data[patient][FEAT_LIST] = batched_data[patient][FEAT_LIST].interpolate(method='linear', 
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
        if batched_data[patient].isnull().values.any():
            it_NaN_batches += 1
            print("Full NaN-batch:", it_NaN_batches)
            points_in_NaN += batched_data[patient].shape[0]
            continue

        # Torch tensor shape: (N, L, H_in)
        #N = n; L = SEQUENCE_LENGTH; H_in = len(FEAT_LIST)
        if n < SEQUENCE_LENGTH:
            feats = batched_data[patient][FEAT_LIST]
            labels = batched_data[patient][LABEL]
            feats = torch.Tensor(feats.values)
            labels = torch.Tensor(labels.values)
            labels = torch.reshape(labels, (len(labels),1))
            
            batch_sequence_feats.append(feats)
            batch_sequence_labels.append(labels)
            if1 += 1

        elif n >= SEQUENCE_LENGTH:
            total += batched_data[patient].shape[0]

            n_modulo = n % SEQUENCE_LENGTH
            n_full_sequences = int((n-n_modulo)/SEQUENCE_LENGTH)
            
            for i in range(n_full_sequences):
                start = i*SEQUENCE_LENGTH
                end = i*SEQUENCE_LENGTH + SEQUENCE_LENGTH
                
                feats = batched_data[patient][FEAT_LIST][start:end]
                labels = batched_data[patient][LABEL][start:end]
                
                feats = torch.Tensor(feats.values)
                labels = torch.Tensor(labels.values)
                labels = torch.reshape(labels, (len(labels),1))
 
                if2 += feats.shape[0]
                batch_sequence_feats.append(feats)
                batch_sequence_labels.append(labels)

            if n-n_modulo != 0:
                feats = batched_data[patient][FEAT_LIST][(i+1)*SEQUENCE_LENGTH:]
                labels = batched_data[patient][LABEL][(i+1)*SEQUENCE_LENGTH:]
                feats = torch.Tensor(feats.values)
                labels = torch.Tensor(labels.values)
                labels = torch.reshape(labels, (len(labels),1))

                batch_sequence_feats.append(feats)
                batch_sequence_labels.append(labels)
                
                if2 += feats.shape[0]
                
        # T: longest sequence
        # B: batch size
        # *: Any number of dimensions
        padded_batch_labels = nn.utils.rnn.pad_sequence(batch_sequence_labels, batch_first=True, padding_value=0.0)
        # Pad sequences:
        padded_batch_feats = nn.utils.rnn.pad_sequence(batch_sequence_feats, batch_first=True, padding_value=0.0)
        sequenced_batches[patient] = (padded_batch_feats, padded_batch_labels)

    sequencify_total_points = 0
    for patient in sequenced_batches.keys():
        sequencify_total_points += sequenced_batches[patient][0].shape[0] * sequenced_batches[patient][0].shape[1]
    print("sequencify_total_points:", sequencify_total_points, "NaN:", points_in_NaN, "dif:", total-points_in_NaN)
    print("points_in_NaN:", points_in_NaN)
    print("if1:", if1)
    print("if2:", if2)
    print("total:", total, "NaN:", points_in_NaN, "sum:", total+points_in_NaN)
    print("sequencify_total_points-total:", sequencify_total_points-total)
    return sequenced_batches

def splitData(sequenced_batches, sepsis_patients, test_size, only_patients=False):
    n_batches = len(sequenced_batches)
    n_sepsis_patients = len(sepsis_patients)
    
    # Get total n patients for split 
    #n_test = int(round(test_size * n_batches))
    #n_train = n_batches - n_test
    
    # Get n patients with sepsis for split
    n_test_sepsis_patients = int(round(test_size * n_sepsis_patients))
    n_train_sepsis_patients = n_sepsis_patients - n_test_sepsis_patients
    
    # List of all patients
    patients = list(sequenced_batches.keys())

    # Remove sepsis patients from data
    for sepsis_patient in sepsis_patients:
        patients.remove(sepsis_patient)
    #print("Patients:", len(patients))
    patients_without_sepsis = patients
    patients_with_sepsis = sepsis_patients
    
    n_healthy = len(patients_without_sepsis)
    n_test_healthy_patients = int(round(test_size * n_healthy))
    n_train_healthy_patients = n_healthy - n_test_healthy_patients
    
    # Get patient splits for healthy and sepsis patients
    train_healthy_patients = patients_without_sepsis[:n_train_healthy_patients]
    test_healthy_patients = patients_without_sepsis[n_train_healthy_patients:]
    train_sepsis_patients = patients_with_sepsis[:n_train_sepsis_patients]
    test_sepsis_patients = patients_with_sepsis[n_train_sepsis_patients:]
    
    """
    print("train_healthy_patients:", len(train_healthy_patients), len(train_healthy_patients)/(n_train_healthy_patients+n_test_healthy_patients))
    print("test_healthy_patients:", len(test_healthy_patients), len(test_healthy_patients)/(n_train_healthy_patients+n_test_healthy_patients))
    print("train_sepsis_patients:", len(train_sepsis_patients), len(train_sepsis_patients)/(n_train_sepsis_patients+n_test_sepsis_patients))
    print("test_sepsis_patients:", len(test_sepsis_patients), len(test_sepsis_patients)/(n_train_sepsis_patients+n_test_sepsis_patients))
    """
    #print("Total:", n_batches, n_sepsis_patients)
    #print("train:", len(train_healthy_patients), len(train_sepsis_patients))
    #print("test:", len(test_healthy_patients), len(test_sepsis_patients))
    
    train_patients = train_healthy_patients + train_sepsis_patients
    test_patients = test_healthy_patients + test_sepsis_patients
    
    random.shuffle(train_patients)
    random.shuffle(test_patients)
    
    if only_patients:
        train = []
        for patient in patients_with_sepsis:
            train.append(sequenced_batches[patient])
        return train
    # Create trainset
    train = []
    for patient in train_patients:
        train.append(sequenced_batches[patient])
    # Create testset
    test = []
    for patient in test_patients:
        test.append(sequenced_batches[patient])

    random.shuffle(train)
    random.shuffle(test)

    print("Train:", len(train), "Test:", len(test), "(train+test)", len(train)+len(test), n_batches)
    
    #"""
    print("--- Randomness check ---")
    total_points = 0
    total_test_points = 0
    total_train_points = 0
    for patient in sequenced_batches.keys():
        #print(batch)
        #raise Exception("TEST")
        total_points += sequenced_batches[patient][0].shape[0] * sequenced_batches[patient][0].shape[1]
    for batch in train:
        total_train_points += batch[0].shape[0] * batch[0].shape[1]
    for batch in test:
        total_test_points += batch[0].shape[0] * batch[0].shape[1]
    print("total_points:", total_points)
    print("total_train_points:", total_train_points)
    print("total_test_points:", total_test_points)
    print("--- Randomness check ---")
    #"""
    return train, test

"""
data = readData(DATA_PATHS[0], raw=False)
data = preprocess(data, print_minmax=True)
#quickPrint(data)
batched_data, sepsis_patient_ids = batchify(data)
sequenced_batches = sequencify(batched_data)
train, test = splitData(sequenced_batches, sepsis_patient_ids, test_size=0.3)
#"""
# sequenced_batches = sequencify(batched_data)


# --- Depricated --- #
#data_tuples = sequenceTuples(data, 1000)
# --- Depricated --- #
