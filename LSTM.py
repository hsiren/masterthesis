# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:17:10 2022

@author: hensir
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:07:56 2022

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
class LSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device="cpu", bidirectional=False):
        super(LSTM, self).__init__()
        
        self.device = device        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        if bidirectional:
            self.D = 2
        else:
            self.D = 1
        
        # RNN-layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)# dropout=0.0)
        # Note: batch_first=True formats output to: (batch, sequence, features)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

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
        c0 = torch.zeros(self.layer_dim, N, h_out).requires_grad_().to(self.device)
        
        # Format data type
        x = x.float()
        h0 = h0.float()
        
        # Forward pass
        out, (h, c) = self.lstm(x, (h0.detach(), c0.detach()) ) #h0.detach()
        
        #out = out[:,-1,:]
        #out = self.fc(out)
        out = self.fc(out).squeeze()
        out = self.sigmoid(out)
        return out