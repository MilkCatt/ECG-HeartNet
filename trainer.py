import os
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import F1Score
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score

# Our objects
from utils import find_optimal_thresholds


class Trainer:
    def __init__(self, model, device, pos_weights=None, accum_steps=4, checkpoint_interval=256, lr=1e-4,
                 resume_checkpoint=None):
        self.model = model
        self.device = device
        self.accum_steps = accum_steps
        self.checkpoint_interval = checkpoint_interval
        
        # Initialize essential components
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.f1 = F1Score(task='multilabel', num_labels=self.model.num_classes, average=None)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device) if pos_weights is not None else None)
        self.accum_loss = 0.0
        self.loss_history = []
        self.acc_history = []
        self.batch_count = 0
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',              # maximize F1
            factor=0.5,              # halve the LR
            patience=2,              # after 2 stagnating checkpoints
            threshold=0.001,
        )

        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        """Load training state from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Essential parameters
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Training progress
        self.loss_history = checkpoint['loss_history']
        self.acc_history = checkpoint['acc_history']
        self.batch_count = checkpoint.get('batch_count', 0)
        self.start_epoch = checkpoint['epoch']
        self.start_batch = checkpoint.get('batch', 0) + 1
        
        # Configurations
        self.checkpoint_interval = checkpoint.get('checkpoint_interval', 
                                                 self.checkpoint_interval)
        
        print(f"Loading epoch {self.start_epoch} batch {self.start_batch}")

    def train(self, train_dataloader, test_dataloader, num_epochs, save_path="training_progress"):
        os.makedirs(save_path, exist_ok=True)
        self.model.train()
        
        for epoch in range(self.start_epoch, num_epochs):
            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                if batch_idx < self.start_batch:
                    continue
                
                
                # Forward pass
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels) / self.accum_steps

                # Backward pass
                loss.backward()

                # Every batch
                self.accum_loss += loss.item()
                self.batch_count += 1
                
                # Every accum_steps
                if (batch_idx + 1) % self.accum_steps == 0:
                    self._update_parameters()
                    
                    # Save loss
                    avg_loss = self.accum_loss
                    self.loss_history.append([self.batch_count, avg_loss])
                    self.accum_loss = 0.0

                    print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_dataloader)} | "
                        f"Avg Loss: {avg_loss:.4f}")

                # Every checkpoint_interval
                if self.batch_count % self.checkpoint_interval == 0:
                    acc = self.evaluate(test_dataloader)
                    self.acc_history.append([self.batch_count, acc])

                    # Scheduler step based on F1 macro average
                    avg_f1 = np.mean(acc["f1_per_class"])
                    self.scheduler.step(avg_f1)

                    self._save_checkpoint(save_path, epoch, batch_idx)
                
                del inputs, labels, outputs, loss
            self.start_batch = 0

    def evaluate(self, dataloader):
        self.model.eval()
        total_samples = 0
        num_classes = self.model.num_classes
        mismatches_per_class = torch.zeros(num_classes, device=self.device)

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())
                total_samples += inputs.size(0)

        # Print for sanity check TODO: Replace with printing accuracy metrics
        print("Example probs:", probs[0].detach().cpu().numpy())
        print("Ground truth :", labels[0].cpu().numpy())

        # Stack across batches
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)

        # Find optimal thresholds
        optimal_thresholds = find_optimal_thresholds(all_labels, all_probs)

        # Apply thresholds
        preds = (all_probs >= optimal_thresholds).astype(int)

        # Metrics
        f1_score_per_class = f1_score(all_labels, preds, average=None, zero_division=0)
        hamming_loss_per_class = np.mean(np.not_equal(preds, all_labels), axis=0)
        overall_hamming_loss = np.mean(np.not_equal(preds, all_labels))

        precision_per_class = precision_score(all_labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, preds, average=None, zero_division=0)

        conf_matrices = multilabel_confusion_matrix(all_labels, preds)
        tp = conf_matrices[:, 1, 1].tolist()
        fp = conf_matrices[:, 0, 1].tolist()
        fn = conf_matrices[:, 1, 0].tolist()
        tn = conf_matrices[:, 0, 0].tolist()

        return {
            "f1_per_class": f1_score_per_class,
            "overall_hamming_loss": overall_hamming_loss,
            "hamming_loss_per_class": hamming_loss_per_class,
            "precision": precision_per_class,
            "recall": recall_per_class,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "optimal_thresholds": optimal_thresholds  # Save this if you want to reuse
        }   
    
    def _update_parameters(self):
        """Update model parameters with gradient clipping"""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _save_checkpoint(self, path, epoch, batch_idx):
        """Save model and training state"""
        checkpoint = {
            'epoch': epoch,
            'batch': batch_idx,
            'batch_count': self.batch_count,
            'checkpoint_interval': self.checkpoint_interval,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'acc_history': self.acc_history
        }
        
        torch.save(checkpoint, f"{path}/checkpoint_ep{epoch}_b{batch_idx}.pt")
        print(f"\nCheckpoint saved at epoch {epoch+1} batch {batch_idx+1}")

        np.save(f"{path}/loss_history.npy", np.array(self.loss_history))
        np.save(f"{path}/acc_history.npy", np.array(self.acc_history))

