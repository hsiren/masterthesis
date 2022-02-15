# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:53:46 2022

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

from params import FEAT_LIST

# Loss fun |  
# MSE
class RNN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device="cpu", bidirectional=False):
        super(RNN, self).__init__()
        
        self.device = device        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        
        if bidirectional:
            self.D = 2
        else:
            self.D = 1
        
        # RNN-layers
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity='tanh', batch_first=True)# dropout=0.0)
        # Note: batch_first=True formats output to: (batch, sequence, features)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        self.softmax = nn.Softmax()

    def forward(self, x):
        #print("--- Forward pass ---")
        """
        N = Batch size
        L = Sequence length
        D = 2 if bidirectional, 1 otherwise
        H_in = input_size
        H_out = hidden_size
        """

        # Initialize hidden-state for first input
        D = self.D; num_layers = self.layer_dim; N = len(x); h_out = self.hidden_dim
        h0 = torch.zeros(self.layer_dim, N, h_out).requires_grad_().to(self.device)

        # Format data type
        x = x.float()
        h0 = h0.float()
        
        # Forward pass
        out, h = self.rnn(x, h0.detach()) #h0.detach()
        
        #if self.single_output:
        #out = out[:,-1,:]
        #out = self.fc(out)
        #else:
        # Reshape outputs for fully connected layer: (batch_size, sequence_len, hidden_size)
        #print("out:", out.shape)
        #out = out[:,-1,:]
        #print("out:", out.shape)
        out = self.fc(out).squeeze()
        #print("out:", out.shape)
        #print("out:", out.shape)
        out = self.sigmoid(out)
        #out = self.softmax(out)
        #print("out:", out.shape)
        #raise Exception("STOP")
        return out

"""
def getClassWeights(y):
    weights = MINORITY_CLASS_WEIGHT * y
    weights[weights == 0.0] = MAJORITY_CLASS_WEIGHT

    #if len(torch.unique(y)) == 2:
        #print("y:", torch.unique(y))
        #print("y:", y)
        #print("Weights:", weights)
        #print("y_orig:", y_orig)
    return weights
"""
"""
def validation(model, test, device):
    with torch.no_grad():
        val_loss = 0.0
        for feats, labels in test:
            
            # --- Putting on device --- #
            feats = feats.to(device)
            labels = labels.to(device)
            # --- Putting on device --- #
            
            y_pred_val = model(feats)

            # Format labels
            y_val = formatLabels(labels, single_output=SINGLE_OUTPUT)

            val_weights = getClassWeights(y_val)
            val_loss_fn = nn.BCELoss(val_weights, reduction="sum")
            vloss = val_loss_fn(y_pred_val, y_val)
            val_loss = val_loss + vloss.item() / feats.shape[0]

    return val_loss/len(test)
"""
"""
# Format labels based on the type of output in the RNN
def formatLabels(labels, single_output=True):
    if single_output:
        label_tuple = torch.max(labels, dim=1)
        return label_tuple[0]
    else:
        return labels[:,:,-1]
"""
"""
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
"""
"""
def train(train, test, model, device):
    print("--- Train ---")
    epochs = EPOCHS
    loss_list = []
    val_loss_list = []
    
    print("Putting model to device...")
    model = model.to(device)
    
    # --- Loss --- #
    #loss_fn = nn.BCELoss(weight=torch.Tensor(CLASS_WEIGHT))
    #loss_fn = nn.BCELoss() # TODO: Motivate
    # --- Loss --- #
        
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    data = train
    print("Len data:", len(data))
    print("batch:", type(data[0]))
    time_start = time()
    #model.initH()
    for epoch in tqdm(range(EPOCHS)):
        val_loss_list.append(validation(model, test, device))
        
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
            y = formatLabels(batch_labels, single_output=SINGLE_OUTPUT)
            #print(SINGLE_OUTPUT, "y:", y.shape)
            
            # Loss computation
            weights = getClassWeights(y)
            loss_fn = nn.BCELoss(weights, reduction="sum")
            loss = loss_fn(y_pred, y)
            train_loss = train_loss + loss.item() / batch_feats.shape[0]

            ### TODO: CHECK ###
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### TODO: CHECK ###

        loss_list.append(train_loss/len(data))
        
    print("Plotting:")
    plt.plot(loss_list, "b")
    plt.plot(val_loss_list, "r")
    plt.show()
    print("Plotted.")
    time_end = time()
    
    print("Total time: " + str(time_end-time_start) + "s")
    return model
"""

"""
# TODO: Accuracy metrics etc.
def main():
    #test()
    #raise Exception("PAUSE")
    # --- Try GPU --- #
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    devicePrint(device)
    # --- Try GPU --- #
    
    print("--- Reading data ---")
    data = readData(DATA_PATHS[0], raw=False)
    data = preprocess(data, print_minmax=True)
    batched_data, sepsis_patient_ids = batchify(data)
    sequenced_batches = sequencify(batched_data)
    #train_set, test_set = splitData(sequenced_batches, sepsis_patient_ids, test_size=0.3)
    train_set = splitData(sequenced_batches, sepsis_patient_ids, test_size=0.3, only_patients=True)
    label_id = 6
    
    print("Len_train:", len(train_set))
    x = train_set[label_id][0]
    y = train_set[label_id][1]
    print("x:", x.shape)
    print("y:", y.shape)
    train_set = [train_set[label_id]]
    #train_set = train_set
    test_set = train_set
    #print("train_set:", train_set[label_id][0].shape)
    print("--- Reading data done ---")

    #TODO:
    #Makes sure all is pushed to the GPU (function in class)

    input_dim = len(FEAT_LIST)
    # Output dim
    output_dim = 1
    # Number of features in the hidden state (window size)
    hidden_dim = HIDDEN_DIM
    # Nr layers:
    layer_dim = LAYER_DIM

    print("Initializing model:")
    model = RNN(input_dim, output_dim, hidden_dim, layer_dim, SINGLE_OUTPUT, device, bidirectional=False)
    
    print("Test synthesizing:")
    model.synthesize(x, y, n=1, round_results=False)    
    
    print("Training start:")
    n = 1
    model = train(train_set, test_set, model, device)
    model.synthesize(x, y, n, round_results=True)
    #model.synthesize(x, y, n, round_results=False)
    #model.synthesize(x, y, n, round_results=True)

    # --- Try GPU --- #
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    devicePrint(device)
    # --- Try GPU --- #
    
    input_dim = 1
    # Output dim
    output_dim = 1
    # Number of features in the hidden state (window size)
    hidden_dim = HIDDEN_DIM
    # Nr layers:
    layer_dim = LAYER_DIM

    print("Initializing model:")
    model = RNN(input_dim, output_dim, hidden_dim, layer_dim, SINGLE_OUTPUT, device, bidirectional=False)
    

#testRNN()
#main()
"""