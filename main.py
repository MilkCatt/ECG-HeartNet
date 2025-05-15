# Dependencies
import torch
from torch.utils.data import DataLoader, random_split

# Our objects
from utils import count_parameters
from dataset import NormalizeECG, ECGDataset
from models import ECGCombined
from trainer import Trainer

## ============================================= ##

# Define device for torch
device = torch.device("cpu")

if torch.mps.is_available():
   print("MPS is available")
   device = torch.device("mps")

# CUDA for Nvidia GPUs
if torch.cuda.is_available():
   print("CUDA is available")
   device = torch.device("cuda")
print(device)

## ===============   Settings  ================= ##
## ============================================= ##

# Meta
diagnoses = "data/diagnoses_balanced.csv"
data_path = "data/ecg_clipped"
save_path = "training_progress/new"
checkpoint_interval = 512

# Training Hyperparameters
add_pos_weights = True
normalize = True
batch_size = 4
accum_steps = 8         # Updates every accum_steps batches
starting_lr = 5e-4      # For resuming, set lr (could be lower) at the resume cell below

# Model Embeddings parameters
d_input = 12
d_model = 128

# Model Transformer parameters
nhead = 4
num_encoder_layers = 2
dim_feedforward = 256

## ============ Load Dataset & Model =========== ##
## ============================================= ##
ecg_dataset = ECGDataset(path=data_path, diagnoses=diagnoses, transform=NormalizeECG() if normalize else None)
pos_weights = ecg_dataset.get_pos_weights() if add_pos_weights else None
num_classes = ecg_dataset.get_num_classes()

train_dataset, test_dataset, val_dataset = random_split(
                                            ecg_dataset, [len(ecg_dataset) - 1000, 500, 500])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = ECGCombined(d_input=d_input, d_model=d_model, num_classes=num_classes, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward).to(device)
print(model)
print("Number of parameters in embeddings")
count_parameters(model.embedding_model)
print("Number of parameters in transformer")
count_parameters(model.transformer)


## ================ Go train =================== ##
## ============================================= ##
trainer = Trainer(model, device, accum_steps=accum_steps, lr=starting_lr, pos_weights=pos_weights, checkpoint_interval=checkpoint_interval)
trainer.train(train_dataloader, test_dataloader, num_epochs=10, save_path=save_path)

