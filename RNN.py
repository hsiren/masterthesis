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

from params import FEAT_LIST, EPOCHS, HIDDEN_DIM, LAYER_DIM, CLASS_WEIGHT


class RNN(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim, layer_dim, bidirectional=False):
        super(RNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        if bidirectional:
            self.D = 2
        else:
            self.D = 1
        
        # RNN-layers
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity='tanh', batch_first=True, dropout=0.0)
        # Note: batch_first=True formats output to: (batch, sequence, features)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # --- Not used --- #
        self.h_prev = None
        self.x_prev = None
        # --- Not used --- #        

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
        h0 = torch.zeros(self.layer_dim, N, h_out).requires_grad_()
        #print("h0:", h0.shape)
        #print("x:", x.shape)
        
      
        # TODO: Check
        # Format data type
        x = x.float()
        h0 = h0.float()
        
        # Forward pass
        out, h0 = self.rnn(x, h0) #h0.detach()
        
        # Save previous state
        self.h_prev = h0
        self.x_prev = x
        
        #print("RNN_OUT:", out.shape)
        #print("RNN_H0:", h0.shape)
        
        # Reshape outputs for fully connected layer: (batch_size, sequence_len, hidden_size)
        out = out[:,-1,:]
        
        # Output
        out = self.fc(out)
        out = self.sigmoid(out)
        
        #print("out:", out.shape)
        #print(out[:10])
        #print("x:")
        #print(x[:10])
        #print("--- Forward ---")
        return out

# Predict gaps in data (NaN-values) (NOT USED)
def predict_gap(sequence, model, one_datapoint=True):
    predicted_out = model.predict_sequence()
    inds = sequence.loc[pd.isnull(sequence).any(1), :].index.values
    all_indices = sequence.index.values.tolist()
    predicted_out = pd.DataFrame(predicted_out.detach().numpy()[0], 
                                 index=all_indices, 
                                 columns=FEAT_LIST)

    # Faster substitution (for SEQUENCE_LENGTH=1)
    if one_datapoint:
        sequence = predicted_out
    else:
        # Substitution for SEQUENC_LENGTH > 1
        for ind in inds:
            # print("ind:", ind, sequence.loc[[ind]])
            sequence.loc[[ind]] = predicted_out.loc[[ind]]
    return sequence

def train(data, model, device):
    print("--- Train ---")
    epochs = EPOCHS
    loss_list = []
    
    print("Putting model to device...")
    model = model.to(device)
    
    # --- Why some losses do not work and some do? --- #
    #loss_fn = nn.CrossEntropyLoss()#weight=torch.Tensor(CLASS_WEIGHT))
    #loss_fn = nn.MSELoss()
    #loss_fn = nn.L1Loss()
    
    #loss_fn = nn.BCELoss(weight=torch.Tensor(CLASS_WEIGHT))
    loss_fn = nn.BCELoss() # TODO: Motivate
    # --- Why some losses do not work and some do? --- #
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    print("Len data:", len(data))
    print("batch:", type(data[0]))
    running_loss = 0.0
    for epoch in tqdm(range(EPOCHS)):
        for batch_feats, batch_labels in data:
            
            # Putting batch to device
            batch_feats.to(device)
            batch_lsbels.to(device)
            
            # Forward pass
            y_pred = model(batch_feats)
            
            # Format labels
            label_tuple = torch.max(batch_labels, dim=1)
            y = label_tuple[0]

            # Loss computation  
            loss = loss_fn(y_pred, y)
            
        
            ### TODO: CHECK ###
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### TODO: CHECK ###
            
            # Save loss
            #running_loss = running_loss + loss.item()
            #loss_list.append(running_loss)
            loss_list.append(loss.item())
            #raise Exception("PAUSE")
            
    print("Plotting:")
    #print("loss_list:", loss_list)
    plt.plot(loss_list)
    plt.show()
    print("Plotted.")

# TODO: Accuracy metrics etc.
def main():
    # --- Try GPU --- #
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    devicePrint(device)
    # --- Try GPU --- #
    
    print("--- Reading data ---")
    data = readData(DATA_PATHS[0], raw=False)
    data = preprocess(data, print_minmax=True)
    quickPrint(data)
    batched_data = batchify(data)
    sequenced_batches = sequencify(batched_data)
    
    print("--- Reading data done ---")

    """
    TODO:
        Makes sure all is pushed to the GPU (function in class)
    """
    
    # Get batch
    #print("n_batches:", len(sequenced_batches))
    #batch0 = sequenced_batches[0]
    #print("len(batch0):", len(batch0))
    #print("n_feats:", batch0[0][0].shape[1], len(FEAT_LIST))

    input_dim = len(FEAT_LIST)
    # Output dim
    output_dim = 1
    # Number of features in the hidden state (window size)
    #hidden_dim = len(FEAT_LIST)
    hidden_dim = HIDDEN_DIM
    # Nr layers:
    layer_dim = LAYER_DIM

    print("Initializing model:")
    model = RNN(input_dim, output_dim, hidden_dim, layer_dim, bidirectional=True)
    
    """
    batch0 = sequenced_batches[0]
    feats0 = batch0[0] # first index: batch tuple, second index: feats or labels
    labels0 = batch0[1]
    print("Batch0:", type(batch0))
    print("Feats0:", feats0.shape)
    print("Labels0:", labels0.shape)
    #output = model(sequence)
    print("Forward pass:")
    #print("sequence:", sequence)
    y_pred = model(feats0)
    print("y_pred0:", y_pred.shape)
    labels0_tuple = torch.max(labels0, dim=1)
    labels0 = labels0_tuple[0]
    print("labels0:", labels0)
    print("labels0:", labels0.shape)
    #plt.plot(labels0)
    #plt.show()
    """
    
    print("Training start:")
    train(sequenced_batches, model, device)

main()
        
"""
# --- Forward pass test --- #
print("data_tuples:", len(data_tuples))
# Number of expected features in the input (test: 3)
input_dim = data_tuples[0][0].shape[1]
print("input_dim:", input_dim)
# Output dim
output_dim = 1
# Number of features in the hidden state (window size)
hidden_dim = 3
# Nr layers:
layer_dim = 1

print("Initializing model:")
model = RNN(input_dim, output_dim, hidden_dim, layer_dim)
print("Running forward pass:")
sequence = data_tuples[0][0] #Check NaN:
print("Sequence:", sequence.shape, sequence)
out = model(sequence)
print("Done!")
# --- Forward pass test --- #
"""     