# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:04:09 2022

@author: hensir
"""
import torch
from torch import nn
#from torch.autograd import gradcheck
import matplotlib.pyplot as plt
import numpy as np
from datahandler import *
import math
from tqdm import tqdm, trange
from time import time
from sklearn.metrics import roc_auc_score
import json
import argparse
import os
import pickle as pkl

from glob import glob

from params import CFG_DIR, FEAT_LIST
from train import train
from datahandler import *
from RNN import RNN
from GRU import GRU
from LSTM import LSTM

SERVER = True

def getParams():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="Input config")
    args = parser.parse_args()
    
    if SERVER:
        outfname = os.path.join("res",
                                os.path.basename(os.path.dirname(args.i)),
                                os.path.basename(args.i).replace(".json", ".pkl"))
        cfg = json.load(open(args.i))
    else:
        cfg = json.load(open(CFG_DIR))
    data_cfgs = cfg["data"]
    model_cfgs = cfg["model"]
    train_cfgs = cfg["train"]
    print("data_cfgs:\n", data_cfgs)
    print("model_cfgs:\n", model_cfgs)
    print("train_cfgs:\n", train_cfgs)
    
    """
    make exp_dir=conv2d singularity exec --nv ../../mycontainer.sif python3 factorialHMM.py -i configs/conv2d/10.json
    """
    return data_cfgs, model_cfgs, train_cfgs, cfg

def synthesize(model, x, y, title_label):
    """
    N = Batch size
    L = Sequence length
    D = 2 if bidirectional, 1 otherwise
    H_in = input_size
    H_out = hidden_size
    """
    plot_list = []
    plot_tensor = None
 
    out = model.forward(x)
    plot_tensor = out.detach()
    y = y[:,:,-1]
    
    # Create plotable list:
    label_list = []
    for i in range(plot_tensor.shape[0]):
        plot_list += plot_tensor[i]
        label_list += y[i]

    fig, ax = plt.subplots(1,1)
    ax.plot(plot_list, "g", label="Prediction")
    ax.plot(label_list, "b", label="Label")
    ax.legend()
    ax.set_title(title_label)
    return fig

def getModel(model, input_dim, hidden_dim, layer_dim, output_dim, device):
    if model == "RNN":
        return RNN(input_dim, hidden_dim, layer_dim, output_dim, device)
    elif model == "GRU":
        return GRU(input_dim, hidden_dim, layer_dim, output_dim, device)
    elif model == "LSTM":
        return LSTM(input_dim, hidden_dim, layer_dim, output_dim, device)
    else: raise Exception("No model specified")

def trainingPrep(train_folds, test_folds, SEQUENCE_LENGTH):
    #print("trainingPrep")
    #"""
    train_set = train_folds
    test_set = test_folds
    
    #print("- train_set:", train_set.shape)
    #print("- test_set:", test_set.shape)
    #print(train_set.shape[0] + test_set.shape[0])
    
    # Process train_set:
    train_set, gap_value = replaceGapsWithNaN(train_set, print_minmax=False)
    #gap_value = 0.0
    #train_set = fillNaNs(train_set, gap_value)
    train_set, scaler = standardizeData(train_set)
    train_set = fillNaNs(train_set, gap_value) # Add gap_value after standardization
    
    # Process test_set:
    test_set, _ = replaceGapsWithNaN(test_set, print_minmax=False)
    #test_set = fillNaNs(test_set, gap_value)
    test_set = scaleTestSet(test_set, scaler)
    test_set = fillNaNs(test_set, gap_value)  # Add gap_value after standardization
    
    # Sequencify train_set:
    batched_train_data, _ = batchify(train_set)
    sequenced_train_batches = sequencify(batched_train_data, SEQUENCE_LENGTH)
    
    train_set = []
    for patient in sequenced_train_batches.keys():
        train_set.append(sequenced_train_batches[patient])
    
    # Sequencify test_set:
    batched_test_data, _ = batchify(test_set)
    sequenced_test_batches = sequencify(batched_test_data, SEQUENCE_LENGTH)
    
    test_set = []
    for patient in sequenced_test_batches.keys():
        test_set.append(sequenced_test_batches[patient])
    #raise Exception("PAUSE")
    #"""
    return train_set, test_set

def main():
    # --- Try GPU --- #
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    devicePrint(device)
    # --- Try GPU --- #
    
    # Get params:
    data_cfgs, model_cfgs, train_cfgs, cfgs = getParams()
    
    # PARAMS #
    if not SERVER:
        SEQUENCE_LENGTH = data_cfgs["SEQUENCE_LENGTH"][0]
        MODEL = model_cfgs["MODEL"][0]
        HIDDEN_DIM = model_cfgs["HIDDEN_DIM"][0]
        LAYER_DIM = model_cfgs["LAYER_DIM"][0]
        LR = train_cfgs["LR"][0]
        EPOCHS = train_cfgs["EPOCHS"][0]
        LOSS = train_cfgs["LOSS"][0]
    else:
        SEQUENCE_LENGTH = data_cfgs["SEQUENCE_LENGTH"]
        MODEL = model_cfgs["MODEL"]
        HIDDEN_DIM = model_cfgs["HIDDEN_DIM"]
        LAYER_DIM = model_cfgs["LAYER_DIM"]
        LR = train_cfgs["LR"]
        EPOCHS = train_cfgs["EPOCHS"]
        LOSS = train_cfgs["LOSS"]
    # PARAMS #
    
    # --- Preprocessing --- #
    train_folds, test_folds = initial_prerpocessing_feat(folds=10)
    # --- Preprocessing --- #
    """
    Note: Data contains currently only patients with sepsis!
    """ 
    input_dim = len(FEAT_LIST)
    hidden_dim = HIDDEN_DIM # Number of features in the hidden state (window size)
    layer_dim = LAYER_DIM # Nr layers
    output_dim = 1

    # Get model:
    model_name = "RNN"
    #model_name = MODEL
    model = getModel(model_name, input_dim, HIDDEN_DIM, LAYER_DIM, output_dim, device)
    
    print("Training start:")
    folds = 2# len(test_folds)
    RESULTS = {"avg_training_time": [], 
               "avg_final_training_loss": [],
               "avg_final_validation_loss": []}
    loss_figs = []
    prediction_figs = []
    for i in range(folds):
        print("--- FOLD: " + str(i+1) + " ---")
        train_set, test_set = trainingPrep(train_folds[i], test_folds[i], SEQUENCE_LENGTH)
                
        model, results = train(train_set, model, device, LR, EPOCHS, LOSS, test_set)
        
        name = "Model_" + str(MODEL) + "_Fold_" + str(i+1) + "," + str(folds) + "_Loss_" + str(LOSS) + "_Model_" + str(MODEL) + "_Layers_" + str(LAYER_DIM) + "_Epocs_" + str(EPOCHS)
        fig, ax = plt.subplots(1,1)
        ax.plot(results["loss_list"], "b", label="Training loss")
        ax.plot(results["val_loss_list"], "r", label="Validation loss")
        title_label = "Model: " + str(MODEL) + " Fold: " + str(i+1) + "/" + str(folds) + " Loss: " + str(LOSS)
        ax.set_title(title_label)
        ax.legend()
        loss_figs.append(fig)
        
        if not SERVER:
            fig.savefig("plots/{}.pdf".format(name))
        
        RESULTS["avg_training_time"].append(results["training_time"])
        RESULTS["avg_final_training_loss"].append(results["final_training_loss"])
        RESULTS["avg_final_validation_loss"].append(results["final_validation_loss"])
        for label_id in range(len(test_set)):
            x = test_set[label_id][0]
            y = test_set[label_id][1]
            prediction_figs.append(synthesize(model, x, y, title_label+" ID: " + str(label_id)))
        
    RESULTS["loss_figs"] = [pkl.dumps(f) for f in loss_figs]
    RESULTS["prediction_figs"] = [pkl.dumps(f) for f in prediction_figs]
    RESULTS["avg_final_training_loss"] = sum(RESULTS["avg_final_training_loss"])/len(RESULTS["avg_final_training_loss"])
    RESULTS["avg_final_validation_loss"] = sum(RESULTS["avg_final_validation_loss"])/len(RESULTS["avg_final_validation_loss"])
    pkl.dump([cfgs, RESULTS], open("test.pkl", "wb+"))

main()