import re
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchmetrics import F1Score
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

class ECGTransformer(nn.Module):
    def __init__(self, d_model, num_classes=63, nhead=8, num_encoder_layers=2, dim_feedforward=2048):
        super().__init__()
        
        # Define encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True)

        # Encoder stack
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        encoded = self.transformer(x)
        # encoded shape: (batch_size, seq_len, d_model)
        # Pick out only the last in the sequence for classification
        encoded = encoded[:, -1, :]
        result = self.classifier(encoded)
        return result
    
class ECGEmbeddings_old(nn.Module):
    def __init__(self, d_input, d_model, n_conv_layers=8, window_size=51):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(d_model if i>0 else d_input, d_model, window_size, stride=1, padding='same')
            for i in range(n_conv_layers)
        ])
        self.activation = nn.ReLU(inplace=False)  # Important for checkpointing

    def forward(self, x):
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if i < len(self.conv_layers) - 1:
                x = self.activation(x)

        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        return x
    
class ECGEmbeddings(nn.Module):
    def __init__(self, d_input, d_model):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = d_input
        out_channels = d_input

        # Dynamically adjust the number of channels to reach d_model
        while out_channels < d_model:
            out_channels = min(d_model, out_channels * 2)  # Double channels, but cap at d_model
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, 1, stride=1, padding='same'))
            in_channels = out_channels

        self.activation = nn.ReLU(inplace=False)  # Important for checkpointing

    def forward(self, x):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i < len(self.conv_layers) - 1:  # Apply activation except for the last layer
                x = self.activation(x)

        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        return x
    
class ECGCombined(nn.Module):
    def __init__(self, d_input, d_model, num_classes=63, nhead=8, num_encoder_layers=2, dim_feedforward=2048):
        super().__init__()
        self.num_classes = num_classes
        
        self.embedding_model = ECGEmbeddings(d_input, d_model)
        self.transformer = ECGTransformer(d_model, num_classes, nhead, num_encoder_layers, dim_feedforward)

    def forward(self, x):
        x = self.embedding_model(x)
        x = x.permute(0, 2, 1)       # Reshape to (batch_size, seq_len, d_model)
        x = self.transformer(x)
        return x