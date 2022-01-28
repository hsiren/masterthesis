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

from params import FEAT_LIST, SINGLE_OUTPUT, EPOCHS, HIDDEN_DIM, LAYER_DIM, LEARNING_RATE, MINORITY_CLASS_WEIGHT, MAJORITY_CLASS_WEIGHT


class RNN(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim, layer_dim, single_output, device="cpu", bidirectional=False):
        super(RNN, self).__init__()
        
        self.device = device        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        if bidirectional:
            self.D = 2
        else:
            self.D = 1
        
        # RNN-layers
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity='tanh', batch_first=True, dropout=0.0)
        # Note: batch_first=True formats output to: (batch, sequence, features)
        
        self.single_output = single_output
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # --- Not used --- #
        self.h_prev = None
        self.x_prev = None
        # --- Not used --- #        

    def synthesize(self, x, y, n, round_results=True):
        """
        N = Batch size
        L = Sequence length
        D = 2 if bidirectional, 1 otherwise
        H_in = input_size
        H_out = hidden_size
        """
        plot_list = []
        plot_tensor = None
        D = self.D; num_layers = self.layer_dim; N = len(x); h_out = self.hidden_dim
        h0 = torch.zeros(self.layer_dim, N, h_out).requires_grad_().to(self.device)
        #x0 = torch.unsqueeze(x, 0)
        x0 = x
        print("x0:", x0.shape)
        print("h0:", h0.shape)
        print("y:", y.shape)
        # is n a sequence or a data point?
        for point in range(n):
            print("Point:", point)
            x0 = x0.float()
            h0 = h0.float()
            
            # Forward pass
            # Outputs: sequence or singular output
            out, h = self.rnn(x0, h0) #h0.detach()
            
            #print("x0:", x0.shape)
            h0 = h
            x0 = out
            #print("out:", out.shape)

            out = out[:,:,-1]
            out = self.sigmoid(out)
            #print("out:", out.shape)#, out)
            plot_tensor = out.detach()
            #out.
            #raise Exception("STOP")
            
        y = y[:,:,-1]
        label_list = []
        for i in range(plot_tensor.shape[0]):
            plot_list += plot_tensor[i]
            #print("plt_tensor:", type(plot_tensor[i]), plot_tensor[i])
            label_list += y[i]
            #print("y:", type(y[i]), y[i])

        if round_results:
            plot_list = np.array(plot_list)
            plot_list = np.round(plot_list).tolist()
        #print("L:", len(plot_list), plot_tensor.shape[0]*plot_tensor.shape[1])
        plt.plot(plot_list, "g")
        plt.plot(label_list, "b")
        plt.show()

    def forward(self, x):
        #print("--- Forward pass ---")
        """
        N = Batch size
        L = Sequence length
        D = 2 if bidirectional, 1 otherwise
        H_in = input_size
        H_out = hidden_size
        """
        D = self.D; num_layers = self.layer_dim; N = len(x); h_out = self.hidden_dim

        # Initialize hidden-state for first input
        # h0: (D*num_layers, N, h_out) --> (1, 1, 5)
        h0 = torch.zeros(self.layer_dim, N, h_out).requires_grad_().to(self.device)
        
        # TODO: Check
        # Format data type
        x = x.float()
        h0 = h0.float()
        
        # Forward pass
        out, h = self.rnn(x, h0) #h0.detach()
        
        if self.single_output:
            out = out[:,-1,:]
            out = self.fc(out)
            out = self.sigmoid(out)
        else:
            # Output
            # Reshape outputs for fully connected layer: (batch_size, sequence_len, hidden_size)
            out = out[:,:,-1]
            out = self.sigmoid(out)
        
        return out

def getClassWeights(y):
    weights = MAJORITY_CLASS_WEIGHT * y
    weights[weights == 0.0] = MINORITY_CLASS_WEIGHT
    #if len(torch.unique(y)) == 2:
        #print("y:", torch.unique(y))
        #print("y:", y)
        #print("Weights:", weights)
        #print("y_orig:", y_orig)
    return weights

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
            #label_tuple = torch.max(labels, dim=1)            
            #y_val = label_tuple[0]
            #y_val = labels[:,:,-1]
            y_val = formatLabels(labels, single_output=SINGLE_OUTPUT)
            #print(SINGLE_OUTPUT, "y_pred_val:", y_pred_val.shape)
            #print(SINGLE_OUTPUT, "y_val:", y_val.shape)

            val_weights = getClassWeights(y_val)
            val_loss_fn = nn.BCELoss(val_weights, reduction="sum")
            vloss = val_loss_fn(y_pred_val, y_val)
            val_loss = val_loss + vloss.item() / feats.shape[0]

    return val_loss/len(test)

def formatLabels(labels, single_output=True):
    if single_output:
        label_tuple = torch.max(labels, dim=1)
        return label_tuple[0]
    else:
        return labels[:,:,-1]

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
    #train_loss = 0.0S
    for epoch in tqdm(range(EPOCHS)):
        val_loss_list.append(validation(model, test, device))
        
        train_loss = 0.0
        for batch_feats, batch_labels in data:
            
            # --- Putting to device --- #
            batch_feats = batch_feats.to(device)
            batch_labels = batch_labels.to(device)
            # --- Putting to device --- #

            #"""
            # Sequence prop:            
            #print("batch_feats:", batch_feats.shape)
            #print("batch_labels:", batch_labels.shape)
            train_loss = 0.0
            for i in range(batch_feats.shape[0]):
                #print("Sample:", batch_feats[i].shape)
                # Forward pass
                y_pred = model(batch_feats)
                
                # Format labels
                y = formatLabels(batch_labels, single_output=SINGLE_OUTPUT)
                #print(SINGLE_OUTPUT, "y:", y.shape)
                
                # Loss computation
                weights = getClassWeights(y)
                #loss_fn = nn.BCELoss(weights, reduction="sum")
                loss_fn = nn.MSELoss()
                loss = loss_fn(y_pred, y)
                train_loss = train_loss + loss.item() / batch_feats[i].shape[0]

                ### TODO: CHECK ###
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ### TODO: CHECK ###
                #raise Exception("PAUSE")
            loss_list.append(train_loss/batch_feats.shape[0])
            #"""
            """
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
            #"""

        loss_list.append(train_loss/len(data))
        
    print("Plotting:")
    #print("loss_list:", loss_list)
    #print("val_loss_list:", val_loss_list)
    plt.plot(loss_list, "b")
    #plt.plot(val_loss_list, "r")
    plt.show()
    print("Plotted.")
    time_end = time()
    
    print("Total time: " + str(time_end-time_start) + "s")
    return model

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
    #quickPrint(data)
    batched_data, sepsis_patient_ids = batchify(data)
    sequenced_batches = sequencify(batched_data)
    #train_set, test_set = splitData(sequenced_batches, sepsis_patient_ids, test_size=0.3)
    train_set = splitData(sequenced_batches, sepsis_patient_ids, test_size=0.3, only_patients=True)
    test_set = [train_set[0]]
    train_set = [train_set[0]]
    print("--- Reading data done ---")

    """
    TODO:
        Makes sure all is pushed to the GPU (function in class)
    """
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
    
    
    print("Training start:")
    model = train(train_set, test_set, model, device)
    n = 1
    x = train_set[0][0]#[0]
    y = train_set[0][1]#[0]
    print("x:", x.shape)
    print("y:", y.shape)
    model.synthesize(x, y, n, round_results=False)
    model.synthesize(x, y, n, round_results=True)

main()