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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupKFold

from params import DATA_FILENAME, DATA_PATHS, FEAT_LIST, LABEL

def readData(file_path, raw=False, prints=False):
    
    if raw:
        print("Reading file:", file_path)
        df = read_compressed_datafile(file_path, nrows=200)
        print("Features:\n", len(df.columns))
        print("Data:", df.shape)
        
        for i,c in enumerate(df.columns):
            if "feats__d" not in c:
                print("i:", i, "c: ["+str(c)+"]")
        
        # 9910 totalt
        
        #print(df.head)
        print("Extracting X & Y...")        
        
        # Each row of F contains a flattened 3d time series + context info
        X = df[[s for s in df.columns if "feats__d" in s]].values
        patient_ids = df[[s for s in df.columns if "group__uid" in s]].values
        print("patient_ids:", patient_ids.shape)
        # X = df[cols].values #Do not use!
        # X = df[[s for s in df.columns if "feats__" in s]].values
        #print("X:", X.shape)
        #x = X[0].reshape((-1, 5))

        Y = df[["target__y"]].values.reshape(-1, 1)
        print("X:", X.shape)
        print("Y:", Y.shape)

        #Z = df.values.reshape(-1,3)
        print("Reshaping...")
        x = X[0].reshape(-1, 3)
        print("X_:", X.shape, x.shape, x.reshape(-1, 3).shape)
        y = Y[0]
        print("Y_:", Y.shape)

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
        plt.show()
        plt.close()
        raise Exception("Raw-stop")
    else:
        df = read_compressed_datafile(file_path)
        if prints:
            print("Reading file:", file_path)
            print("Features:\n", len(df.columns))
            for col in df.columns:
                print(col)
            print("Data:\n", df.head)
            print(df.shape)
            #df[["feats__btb_mean","feats__rf_mean","feats__rf_std","target__y",]].iloc[1:100].plot.line(subplots=True)
            print(df["target__y"].unique())
            print("Group__uid:\n", len(df["group__uid"].unique()))#, df["group__uid"].unique())
    return df

# TODO: Triple check with plots etc...
def replaceGapsWithNaN(data, print_minmax=True):
    replacement_value=math.nan
    gap_value = data["feats__btb_mean"].min()
    data = data.replace(data["feats__btb_mean"].min(), replacement_value)
    data = data.replace(data["feats__rf_mean"].min(), replacement_value)
    
    # Check for columns with gaps
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

    #print(data["target__y"].isnull().values.any())
    #print(data["group__uid"].isnull().values.any())
    #raise Exception("Hello")
    return data, gap_value

def standardizeData(data):
    """
    NOTE: The standard scaler ignores NaN-values.
    """
    # Get features from data
    #prescaled_data = data[FEAT_LIST]
    #labels = data[LABEL]

    scaler = StandardScaler()
    scaler = scaler.fit(data[FEAT_LIST])
    data[FEAT_LIST] = scaler.transform(data[FEAT_LIST])
    return data, scaler

def scaleTestSet(test_set, scaler):
    test_set[FEAT_LIST] = scaler.transform(test_set[FEAT_LIST])
    return test_set

def removeFirst(data):
    #print("removeFirst")
    orig_shape = data.shape
    data = data.drop(0, axis=0)
    data = data.reset_index(drop=True)
    new_shape = data.shape
    #print("- Verified: " + "[" + str(orig_shape[0]-1==new_shape[0]) +"]")
    return data

def removeNaNPatients(data):
    #print("removeNaNPatients")
    #print("- Data shape:", data.shape)

    it_NaN_batches = 0
    patients = data["group__uid"].unique()
    NaN_patients = []
    NaN_batch_found = False
    total_points_in_NaN = 0
    
    ### Verification ###
    #"""
    orig_data_size = data.shape[0]
    #"""
    ### Verification ###
    
    for patient in patients:
        patient_data = data.loc[data['group__uid'] == patient]
        patient_data = patient_data[FEAT_LIST]
        for col in patient_data:
            unique = patient_data[col].unique().tolist()
            if len(unique) == 1 and math.isnan(unique[0]):
                # full NaN-col found! (remove full patient)
                NaN_batch_found = True
                

                total_points_in_NaN += len(patient_data[col].tolist())
                break
        
        if NaN_batch_found:
            it_NaN_batches += 1
            NaN_patients.append(patient)
            NaN_batch_found = False

    #print("- NaN_patients:", it_NaN_batches)
    #print("- Total points to remove:", total_points_in_NaN)
    # remove rows with NaN_patient id:
    for NaN_patient in NaN_patients:
        data = data[data["group__uid"] != NaN_patient]
        #data = data.drop(data.loc[data['group__uid'] == NaN_patient].index)#, inplace=True)

    # Reset indices of the dataframe
    data = data.reset_index(drop=True)
    
    ### Verification ###
    """
    print("- New data shape:", data.shape)
    print("- Verified: " + "[" + str(orig_data_size-data.shape[0]==total_points_in_NaN) +"]")
    #"""
    ### Verification ###
    
    return data

def interpolate(data, THRESHOLD):
    print("interpolate")
    patients = data["group__uid"].unique()
    for patient in tqdm(patients):
        data.loc[data['group__uid'] == patient] = data.loc[data['group__uid'] == patient].interpolate(method='linear', 
                                                                                                      axis=0, # 0: (column)
                                                                                                      limit=THRESHOLD,
                                                                                                      inplace=False, 
                                                                                                      limit_direction="both", 
                                                                                                      limit_area=None, 
                                                                                                      downcast=None)
    return data

def removeRestNaNs(data):
    data = data.dropna(axis=0, how="any")
    return data

def devicePrint(device):
    mode = None
    if device == "cuda":
        mode = "GPU"
        print("===== {Running on} =====")
        print("=====    {"+device+"}    =====")
        print("===== {Running on} =====")
    else:
        mode = "CPU"
        print("===== {Running on} =====")
        print("=====    {"+device+"}     =====")
        print("===== {Running on} =====")

def batchify(data):
    """
    Note:
        - Creates batches from data where each batch represents one patient
        - Checks whether batch/patient only contains NaN-values and removes said batch
    """
    #print("batchify")
    #print("- n_'Group__uid':", len(data["group__uid"].unique()))
    patients = data["group__uid"].unique()
    patient_batches = {}
    sepsis_patients = []
    total_points = 0
    for patient in patients:
        patient_data=data.loc[data['group__uid'] == patient]
        vc = patient_data["target__y"].value_counts()
        if vc.shape[0] == 2:
            sepsis_patients.append(patient)
            #print("- Points/patient (sepsis):", patient_data.shape[0], "0:", vc[0], "1:", vc[1])
            """
            x = np.arange(patient_data["target__y"].shape[0])
            plt.plot(x, patient_data["target__y"])
            plt.title(patient)
            plt.show()
            #"""
            #print("---------------------------------------------")

        else:
            #print("- Points/patient:", patient_data.shape[0])
            pass
        total_points += patient_data.shape[0]

        patient_batches[patient] = patient_data
    #print("- n_patients:", len(patient_batches))
    #print("- n_total_points:", total_points)
    return patient_batches, sepsis_patients

def sequencify(batched_data, SEQUENCE_LENGTH):
    """
    NOTE:
        - Dynamically sequencifies batches based on fixed SEQUENCE_LENGTH parameter.
        - Pads overlapping values with zeros!
    """
    sequenced_batches = {}
    n_batches = len(batched_data)
    #print("n_batches:", n_batches, len(batched_data.keys()))
    total = 0
    it_NaN_batches = 0
    points_in_NaN = 0
    m = 0
    #for patient in tqdm(batched_data.keys()):
    for patient in batched_data.keys():
        n = len(batched_data[patient])
        batch_sequence_tuples = []
        batch_sequence_feats = []
        batch_sequence_labels = []
        
        # Verify that no NaNs exist in data.
        if batched_data[patient].isnull().values.any():
            raise Exception("NaN found! (should not have)")

        # Torch tensor shape: (N, L, H_in)
        #N = n; L = SEQUENCE_LENGTH; H_in = len(FEAT_LIST)
        # If the batch is smaller than the SEQUENCED_LENGTH, use max size.
        if n < SEQUENCE_LENGTH:
            feats = batched_data[patient][FEAT_LIST]
            labels = batched_data[patient][LABEL]
            feats = torch.Tensor(feats.values)
            labels = torch.Tensor(labels.values)
            labels = torch.reshape(labels, (len(labels),1))
            
            batch_sequence_feats.append(feats)
            batch_sequence_labels.append(labels)

        # If batch is larger than the SEQUENCES_LENGTH, chop into pieces.
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

        # T: longest sequence
        # B: batch size
        # *: Any number of dimensions
        padded_batch_labels = nn.utils.rnn.pad_sequence(batch_sequence_labels, batch_first=True, padding_value=0.0)
        # Pad sequences:
        padded_batch_feats = nn.utils.rnn.pad_sequence(batch_sequence_feats, batch_first=True, padding_value=0.0)
        sequenced_batches[patient] = (padded_batch_feats, padded_batch_labels)
        
    # Verification
    """
    for key in sequenced_batches.keys():
        batch_feats, batch_labels = sequenced_batches[key]
        print("batch_feats", batch_feats.shape)
        print("batch_feats_final", batch_feats[-1])
        print("batch_labels", batch_labels.shape)
        print("batch_labels_final", batch_labels[-1])
        raise Exception("STOP")
    """
    sequencify_total_points = 0
    for patient in sequenced_batches.keys():
        sequencify_total_points += sequenced_batches[patient][0].shape[0] * sequenced_batches[patient][0].shape[1]
    #print("- sequencify_total_points:", sequencify_total_points, "NaN:", points_in_NaN)
    return sequenced_batches

def getOnlyTargetPatients(sequenced_batches, sepsis_patients):
    """
    Note: 
        - Extracts only patients with sepsis and returns them.
        - No splitting is done.
    """
    sequenced_sepsis_patients = []
    for patient in sepsis_patients:
        sequenced_sepsis_patients.append(sequenced_batches[patient])
    return sequenced_sepsis_patients
    
def splitData(sequenced_batches, sepsis_patients, test_size, only_patients=False, shuffle=False):
    n_batches = len(sequenced_batches)
    n_sepsis_patients = len(sepsis_patients)
    
    # Get n patients with sepsis for split
    n_test_sepsis_patients = int(round(test_size * n_sepsis_patients))
    n_train_sepsis_patients = n_sepsis_patients - n_test_sepsis_patients
    
    # List of all patients
    patients = list(sequenced_batches.keys())

    # Remove sepsis patients from data
    print("Sepsis_patients:", sepsis_patients)
    for sepsis_patient in sepsis_patients:
        print("Sepsis_patients:", sepsis_patient)
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
    #print("Total:", n_batches, n_sepsis_patients)
    #print("train:", len(train_healthy_patients), len(train_sepsis_patients))
    #print("test:", len(test_healthy_patients), len(test_sepsis_patients))
    """
    
    train_patients = train_healthy_patients + train_sepsis_patients
    test_patients = test_healthy_patients + test_sepsis_patients
    
    if shuffle:
        random.shuffle(train_patients)
        random.shuffle(test_patients)
    
    # If only patients with sepsis is requested
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

    if shuffle:
        random.shuffle(train)
        random.shuffle(test)

    print("Train:", len(train), "Test:", len(test), "(train+test)", len(train)+len(test), n_batches)
    return train, test

def getClassWeights(data, sepsis_patient_ids):
    n_samples = 0
    #print(data["target__y"].head(5))
    classes = np.array([0., 1.])
    y = data["target__y"].to_numpy()
    #print("y:", y.shape)
    #weights = compute_class_weight("balanced", classes, y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    #print("weights:", weights)
    class_weights = {"0": weights[0], "1": weights[1]}
    return class_weights
    
def fillNaNs(data, gap_value):
    data = data.fillna(gap_value, axis=0)
    return data

def splitSepsisData(data, test_split_ratio=0.1):
    """
    Note:
        - Extracts patients with sepsis from non-sepsis data.
        - test_split_ratio: Sets n patients into test set. 10% by default (1 patient)
    """
    #print("splitSepsisData")
    patients = data["group__uid"].unique()
    sepsis_patients = []
    frames = []
    for patient in patients:
        patient_data=data.loc[data['group__uid'] == patient]
        
        vc = patient_data["target__y"].value_counts()
        if vc.shape[0] == 2 or 1.0 in vc: # fix
            sepsis_patients.append(patient)
            frames.append(patient_data)
            #print("- Points/patient (sepsis):", patient_data.shape[0], "0:", vc[0], "1:", vc[1])
            """
            x = np.arange(patient_data["target__y"].shape[0])
            plt.plot(x, patient_data["target__y"])
            plt.title(patient)
            plt.show()
            #"""
            #print("---------------------------------------------")
    
    full_data = pd.concat(frames)
    cols = ["group__uid", "target__y"] + FEAT_LIST
    featurized_data = full_data[cols]
    return featurized_data, sepsis_patients

def foldSepsisData(data, sepsis_patients, folds=None):
    """
    Note:
        - Splits data into folds.
        - if folds=None, leave-one-out cross validation will be used
    """
    #print("foldSepsisData")
    train_data_list = []
    test_data_list = []
    
    if folds is None: folds = len(sepsis_patients)
    elif folds > len(sepsis_patients): raise Exception("Too many folds! Folds: " + str(folds) + "and N-sepsis patients: " + str(len(sepsis_patients)))
    else: folds = folds

    var = data[:10] #
    group_kfold = GroupKFold(n_splits=folds)
    n_groups = len(sepsis_patients)
    group_kfold.get_n_splits(sepsis_patients, y=None, groups=sepsis_patients)
    
    sepsis_patients_df = pd.DataFrame(sepsis_patients)
    for train, test in group_kfold.split(X=sepsis_patients, y=None, groups=sepsis_patients):
        train_frames = []
        test_frames = []

        train_groups = sepsis_patients_df.iloc[train].to_numpy().T.tolist()[0]
        test_groups = sepsis_patients_df.iloc[test].to_numpy().T.tolist()[0]
        
        for train_patient in train_groups:
            train_frames.append(data.loc[data['group__uid'] == train_patient])
        train_data = pd.concat(train_frames)
        
        for test_patient in test_groups:
            test_frames.append(data.loc[data['group__uid'] == test_patient])
        test_data = pd.concat(test_frames)
        
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        
    return train_data_list, test_data_list

# --- NOT USED --- #  
def HfoldSepsisData(data, sepsis_patients):
    print("foldSepsisData")
    #print("data:", data.shape)
    #print("Sepsis_patients:", len(sepsis_patients))
    train_data_list = []
    test_data_list = []
    copy_sepsis_patients = sepsis_patients
    for i, patient in enumerate(sepsis_patients):
        frames = []
        #batched_train_data = {}
        #batched_test_data = {}

        copy_sepsis_patients = sepsis_patients[:]
        copy_sepsis_patients.remove(patient)
        train_patients = copy_sepsis_patients
        test_patient = patient

        #print("Test lens:", len(train_patients), len([test_patient]))
        for patient2 in train_patients:
            #batched_train_data[patient2] = data.loc[data['group__uid'] == patient2]
            frames.append(data.loc[data['group__uid'] == patient2])
        
        train_data = pd.concat(frames)
        test_data = data.loc[data['group__uid'] == test_patient]
        
        train_data_list.append(train_data)
        test_data_list.append(test_data)
    
    #print(len(train_data_list), len(test_data_list))
    return train_data_list, test_data_list
# --- NOT USED --- #  

def initial_prerpocessing_feat(folds=None):
    """
    NOTE:
        To be called only once.
    """
    #print("--- initial_prerpocessing_feat ---")
    data = readData(DATA_PATHS[0], raw=False)
    data = removeFirst(data)

    #data, gap_value = replaceGapsWithNaN(data, print_minmax=False)
    #data = removeNaNPatients(data)
    #data = fillNaNs(data, gap_value)

    data, sepsis_patients = splitSepsisData(data)
    train_folds, test_folds = foldSepsisData(data, sepsis_patients, folds)
    
    #data = interpolate(data)
    #data = removeRestNaNs(data)
    
    #data.to_pickle(DATA_FILENAME)
    return train_folds, test_folds
    #print("--- initial_prerpocessing_feat [DONE] ---")

def preprocessing_pipeline_feat(THRESHOLD, SEQUENCE_LENGTH):
    print("--- preprocessing_pipeline_feat ---")
    try:
        data = pd.read_pickle(DATA_FILENAME)
    except FileNotFoundError:
        initial_prerpocessing_feat(THRESHOLD)
        data = pd.read_pickle(DATA_FILENAME)
    
    batched_data, sepsis_patient_ids = batchify(data)
    sequenced_batches = sequencify(batched_data, SEQUENCE_LENGTH)
    class_weights = getClassWeights(data, sepsis_patient_ids)
    sequenced_data = getOnlyTargetPatients(sequenced_batches, sepsis_patient_ids)
    #data = splitData(sequenced_batches, sepsis_patient_ids, test_size=0.3)
    print("--- preprocessing_pipeline_feat [DONE] ---")
    return sequenced_data, class_weights

#initial_prerpocessing_feat()
#preprocessing_pipeline_feat(60, 10)
