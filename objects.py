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

class NormalizeECG:
    def __call__(self, tensor):
        # Z-score normalization per lead
        means = tensor.mean(dim=1, keepdim=True)
        stds = tensor.std(dim=1, keepdim=True)
        return (tensor - means) / (stds + 1e-8)
    
class ECGDataset(Dataset):
    def __init__(self, path="data/ecg", diagnoses='data/diagnoses.csv', transform=None):
        # Load and prepare labels
        self.labels_df = pd.read_csv(diagnoses)

        self.labels_df['ID'] = self.labels_df['ID'].astype(str).str.replace(r'\D', '', regex=True) # Remove the JS
        self.labels_df.set_index('ID', inplace=True)
        self.num_classes = self.labels_df.shape[1]
        print(f'Number of classes: {self.num_classes}')

        self.transform = transform
        self.cache = {}

    def get_pos_weights(self, device = "cpu"):
        # Compute counts
        pos_counts = self.labels_df.sum()
        neg_counts = len(self.labels_df) - pos_counts

        # Calculate pos_weight = #neg / #pos for each class
        pos_weight = (neg_counts / pos_counts).values

        # Move to device
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device= device)
        return pos_weight_tensor
    
    def get_num_classes(self):
        return self.num_classes

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        try:
            # Access the row through iloc of the index,
            # Use the ID to make filepath
            ID = self.labels_df.iloc[idx].name

            file_path = f'data/ecg/{ID}.csv'
            
            # Load ECG data
            df = pd.read_csv(file_path)
            ecg_data = df.drop(columns=['time']).values
            tensor = torch.tensor(ecg_data, dtype=torch.float32).T  # (leads, timesteps)
            
            if self.transform:
                tensor = self.transform(tensor)
                
            # Get corresponding label

            label_values = self.labels_df.loc[ID].values  # Get all label columns
            label = torch.tensor(label_values, dtype=torch.float32)  # Use float for multi-label

            return tensor, label
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            # Return zero tensor and -1 label placeholder
            return torch.zeros((12, 5000), dtype=torch.float32), torch.full((self.num_classes,), -1, dtype=torch.float32)
      