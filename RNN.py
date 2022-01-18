# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 12:53:46 2022

@author: hensir
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from datahandler import *
import math
from tqdm import tqdm

class RNN(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim, layer_dim):
        super(RNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # RNN-layers
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity='tanh', batch_first=True, dropout=0.0)
        # Note: batch_first=True formats output to: (batch, sequence, features)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        self.h_prev = None
        self.x_prev = None
    
    def predict(self):#, x):
        
        h = self.h_prev
        x = self.x_prev
        #x = x.float()

        out, h0 = self.rnn(x, h) #h0.detach()
        print("P_out:", out.shape)        
        # Reshape outputs for fully connected layer: (batch_size, sequence_len, hidden_size)
        #out = out[:,-1,:]
        
        # Output
        #out = self.fc(out)
        #out = self.sigmoid(out)
        
        return out       

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
        # h0: (D*num_layers, N, h_out) --> (1, 1, 5)
        # Note: Am sing one batch (sequence len: 9) for input.
        h0 = torch.zeros(self.layer_dim, 1, self.hidden_dim).requires_grad_()
        #print("h0:", h0.shape)
                
        # Turn input to tensor with shape: (N, L, H_in) --> (1, 9, 3) (with short features)    
        x = torch.tensor(x.values)[None,:]#["feats__btb_mean", "feats__rf_mean", "feats__spo2_mean"]
        #print("x:", x.shape)
        #x = torch.tensor(x.values)#["feats__btb_mean", "feats__rf_mean", "feats__spo2_mean"]

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
        
        #print("out:", out.shape, out[0][0])
        return out
        
def train(data, model):
    print("--- Train ---")
    #data = [data[-1]] # Remove
    epochs = 2
    loss_list = []
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print("Len data:", len(data))
    for i in range(epochs):
        print("Epoch:", i+1)
        for sequence, labels in tqdm(data):
            
            # Skip first value
            sequence = sequence[1:]
            labels = labels[1:]
                        
            #print("Sequence:", sequence)
            # Handle NaN (gaps): Either by skipping/deleting or predicting
            """
            - Skipping/Deletion:
                - 
            """
            """
            - Synthesizing:
                - Attempt with one data point
            """
            # Check for NaN:
            # If contains NaN:
                # Get indices for NaN-vals:
                    # Do prediction on previous datapoint:
                        # How many points and does it matter?
                        # Add predicted datapoint to sequence:
                            # Loop for all NaN-values
            """
            print("Sequence:",  type(sequence))
            # NaN_inds = sequence["feats__btb_mean"].index[sequence["feats__btb_mean"].apply(np.isnan)] 
            NaN_inds = sequence.isnull().any(axis=1)
            NaN_inds_list = NaN_inds.tolist()
            NaN_inds = sequence[sequence['feats__btb_mean'].isnull()].index.tolist()
            # Note: Hard coded single column. Assumption: missing value in 1 column means missing values in all columns.
            print("NaN_inds:\n", NaN_inds, NaN_inds_list)
            synthesized_out = model.predict()
            synthesized_out = synthesized_out.detach().numpy()
            print("Synthesized output:\n", synthesized_out.shape)
            synthesized_out = pd.DataFrame(synthesized_out[0])
            print("Synthesized output:\n", synthesized_out.shape, synthesized_out)
            #sequence = sequence.mask(NaN_inds, synthesized_out)
            test = pd.DataFrame(NaN_inds_list, index=[202,203,204,205,206,207,208,209,210], columns=["a"])
            #print("test:", test.columns, test)
            #keys = ['a','b','c']
            #for k in keys:
            #    test[k] = test['a']
            #print("test:", test)
            #print("test:", test.shape, "Sequence:", sequence.shape, "synthesized_out:", synthesized_out.shape)
            sequence = sequence.mask(sequence == np.nan, synthesized_out)
            print(sequence.isnull())
            sequence = sequence.replace(to_replace = np.nan, value = synthesized_out)
            print("Sequence:", sequence)
            #sequence.where(test, synthesized_out)
            # Replace NaN
            #print("TEST:", sequence.iloc[NaN_inds])
            raise Exception("Pause")
            """
            
            # Forward pass
            y_pred = model(sequence)
            #print("y_pred:", y_pred.shape)
            
            # Loss computation
            labels = torch.tensor(labels.values)#[None,:]
            # Aggregate label
            """
            How to aggregate label?
            Options:
                Sensitive: If label vector contains 1 --> 1
                Mean rounded: Take the mean of the label vector and round.
            """
            # Sensitive
            labels = torch.max(labels)
            labels = torch.reshape(labels, (1,1))
            
            #print("LABELS:", labels.shape, labels)
            #print("Y_PRED:", y_pred.shape, y_pred)
            loss = loss_fn(y_pred, labels)
            
            # Save loss
            loss_list.append(loss.detach())
            
            ### TODO: CHECK ###
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### TODO: CHECK ###
            #raise Exception("PAUSE")
    #print("Plotting:")
    #print("loss_list:", loss_list)
    #plt.plot(loss_list)
    #plt.show()
    #print("Plotted.")

# TODO: Handle NaN-values (gaps in the data)

def main():
    print("--- Reading data ---")
    data = readData(DATA_PATHS[0])
    data = preprocess(data)
    data_tuples = sequenceTuples(data, n_sequences=1000)
    print("--- Reading data done ---")

    # --- Try GPU --- #
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    print("Using " + device + " device")
    # --- Try GPU --- #
    
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
    sequence = data_tuples[0][0][1:] #Check NaN:
    print("Sequence:", sequence.shape, sequence)
    out = model(sequence)
    print("Done!")
    
    print("Training start:")
    #print("Data_tuples:", data_tuples[-1])
    train(data_tuples, model)
    
    

main()
        
        