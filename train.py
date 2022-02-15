# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:58:23 2022

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

from params import FEAT_LIST#, MINORITY_CLASS_WEIGHT, MAJORITY_CLASS_WEIGHT

# TODO: Make according to sklearn
def getClassWeights(y, class_weights):
    minority_class_weight = class_weight["1"]
    majority_class_weight = class_weight["0"]
    weights = minority_class_weight * y
    weights[weights == 0.0] = majority_class_weight
    """
    if len(torch.unique(y)) == 2:
        print("y:", torch.unique(y))
        print("y:", y)
        print("Weights:", weights)
        print("y_orig:", y_orig)
    """
    return weights

def validation(model, test, device, val_loss_fn):
    with torch.no_grad():
        val_loss = 0.0
        for feats, labels in test:
            
            # --- Putting on device --- #
            feats = feats.to(device)
            labels = labels.to(device)
            # --- Putting on device --- #
            
            y_pred_val = model(feats)

            # Format labels
            y_val = formatLabels(labels)

            #val_weights = getClassWeights(y_val)
            #val_loss_fn = nn.BCELoss(val_weights, reduction="sum")
            vloss = val_loss_fn(y_pred_val, y_val)
            val_loss = val_loss + vloss.item() / feats.shape[0]

    return val_loss#/len(test)

# Format labels based on the type of output in the RNN
def formatLabels(labels, single_output=False):
    if single_output:
        label_tuple = torch.max(labels, dim=1)
        return label_tuple[0]
    else:
        return labels[:,:,-1]

# --- NOTE: Not used --- #
def sequentialUpdate(batch_feats, batch_labels, model, optimizer, loss_list):
    # Sequence prop:
    train_loss = 0.0
    #model.initH()
    for i in range(batch_feats.shape[0]):
        # Forward pass
        x_in = torch.unsqueeze(batch_feats[i], dim=0)
        y_pred = model(x_in)

        # Format labels
        y_in = torch.unsqueeze(batch_labels[i], dim=0)
        y = formatLabels(y_in, single_output=SINGLE_OUTPUT)
        
        # Loss computation
        weights = getClassWeights(y)
        loss_fn = nn.BCELoss(weights, reduction="sum")
        #loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y)
        train_loss = train_loss + loss.item() / batch_feats[i].shape[0]

        ### TODO: CHECK ###
        optimizer.zero_grad()
        loss.backward()#retain_graph=True)
        optimizer.step()
        ### TODO: CHECK ###

    loss_list.append(train_loss/batch_feats.shape[0])
    return loss_list
# --- NOTE: Not used --- #

def train(train, model, device, LR, EPOCHS, LOSS, test_set=None, class_weights=None):
    #print("--- Train ---")
    results = {}
    
    epochs = EPOCHS
    loss_list = []
    val_loss_list = []
    
    print("Putting model to device...")
    model = model.to(device)
    
    # --- Loss --- #
    #loss_fn = nn.BCELoss(weight=torch.Tensor(CLASS_WEIGHT))
    if LOSS == "MSE":
        loss_fn = nn.MSELoss(reduction="sum")
        #print("Loss: MSELoss")
    elif LOSS == "BCE":
        loss_fn = nn.BCELoss()
        #print("Loss: BCELoss")
    else:
        #print("Loss:", "Weighted BCELoss")
    # --- Loss --- #
        
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    data = train
    #print("Len data:", len(data))
    #print("batch:", type(data[0]))
    time_start = time()
    #model.initH()
    for epoch in tqdm(range(EPOCHS)):
        val_loss_list.append(validation(model, test_set, device, loss_fn))
        
        train_loss = 0.0
        #model.initH(bdim=batch_feats.shape[0])
        for batch_feats, batch_labels in data:

            # --- Putting to device --- #
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device)
            # --- Putting to device --- #

            # Forward pass
            y_pred = model(batch_feats)
            
            # Format labels
            y = formatLabels(batch_labels)
            
            # Loss computation
            # TODO: Check weight computations correctly.
            #if LOSS == "WBCE:":
            #weights = getClassWeights(y, class_weights)
            #loss_fn = nn.BCELoss(weights, reduction="sum")
            loss = loss_fn(y_pred, y)
            train_loss = train_loss + loss.item() / batch_feats.shape[0]

            ### TODO: CHECK ###
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### TODO: CHECK ###

        loss_list.append(train_loss/len(data))
        

    time_end = time()
    
    print("Total time: " + str(round(time_end-time_start, 2)) + "s")
    print("Final train_loss:", loss_list[-1])
    results["training_time"] = time_end-time_start
    results["loss_list"] = loss_list
    results["final_training_loss"] = loss_list[-1]
    results["model"+str(EPOCHS)+"Epochs"] = model
    results["val_loss_list"] = val_loss_list
    results["final_validation_loss"] = val_loss_list[-1]
    return model, results
