import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from preprocess import SlumDatasetPreprocessor
import os
import gc
from torchvision.models import resnet18
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class SlumDataset(Dataset):
    def __init__(self, images, labels, preprocessor, transform=None):
        self.images = images
        self.labels = labels
        self.preprocessor = preprocessor
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            # Get image and preprocess it
            img = self.images[idx]
            img = self.preprocessor.preprocess_image(img)
            
            # Create binary mask for slum areas
            mask = self.preprocessor.create_binary_mask(self.labels[idx])
            
            # Convert to tensors
            img = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC to CHW
            mask = torch.from_numpy(mask).float()  # Keep as 2D tensor
            
            # Apply transforms if any
            if self.transform:
                img = self.transform(img)
            
            return img, mask
        except Exception as e:
            print(f"Error processing image at index {idx}: {str(e)}")
            # Return a dummy sample in case of error
            return torch.zeros((3, 256, 256)), torch.zeros((256, 256))

class SlumDetector:
    def __init__(self):
        # Initialize model architecture
        self.model = self._create_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def _create_model(self):
        """Create a U-Net style model for segmentation"""
        class UNet(nn.Module):
            def __init__(self, in_channels):
                super(UNet, self).__init__()
                
                # Encoder
                self.enc1 = self._make_layer(in_channels, 64)
                self.enc2 = self._make_layer(64, 128)
                self.enc3 = self._make_layer(128, 256)
                self.enc4 = self._make_layer(256, 512)
                
                # Decoder
                self.dec4 = self._make_layer(512 + 256, 256)
                self.dec3 = self._make_layer(256 + 128, 128)
                self.dec2 = self._make_layer(128 + 64, 64)
                self.dec1 = nn.Sequential(
                    nn.Conv2d(64 + in_channels, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 1, kernel_size=1),
                    nn.Sigmoid()
                )
                
                self.pool = nn.MaxPool2d(2)
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
            def _make_layer(self, in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                # Encoder
                enc1 = self.enc1(x)
                enc2 = self.enc2(self.pool(enc1))
                enc3 = self.enc3(self.pool(enc2))
                enc4 = self.enc4(self.pool(enc3))
                
                # Decoder with skip connections
                dec4 = self.dec4(torch.cat([self.upsample(enc4), enc3], dim=1))
                dec3 = self.dec3(torch.cat([self.upsample(dec4), enc2], dim=1))
                dec2 = self.dec2(torch.cat([self.upsample(dec3), enc1], dim=1))
                dec1 = self.dec1(torch.cat([dec2, x], dim=1))
                
                return dec1
        
        return UNet(in_channels=7)  # 7 channels: 3 for RGB + 4 for features
    
    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print("Starting training...")
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_preds = []
            train_targets = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                
                # Ensure output and target have the same shape
                output = output.squeeze(1)  # Remove channel dimension to match target
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Store predictions and targets for metrics
                pred_mask = (output > 0.5).float()
                train_preds.extend(pred_mask.detach().cpu().numpy().flatten())
                train_targets.extend(target.cpu().numpy().flatten())
            
            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_targets, train_preds)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    output = output.squeeze(1)  # Remove channel dimension
                    val_loss += criterion(output, target).item()
                    
                    pred_mask = (output > 0.5).float()
                    val_preds.extend(pred_mask.cpu().numpy().flatten())
                    val_targets.extend(target.cpu().numpy().flatten())
            
            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_targets, val_preds)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Generate and save confusion matrix every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_confusion_matrix(val_targets, val_preds, epoch + 1)
        
        # Plot final metrics
        self.plot_training_metrics()
    
    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_epoch_{epoch}.png')
        plt.close()
    
    def plot_training_metrics(self):
        # Plot loss curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
    
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                preds = (output.cpu().numpy() > 0.5).astype(int)
                all_preds.extend(preds)
                all_targets.extend(target.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        
        # Generate final confusion matrix
        self.plot_confusion_matrix(all_targets, all_preds, 'final')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

def main():
    try:
        # Initialize detector
        detector = SlumDetector()
        
        # Train the model
        history, test_metrics = detector.train(
            num_epochs=50,
            batch_size=4,  # Reduced batch size to avoid memory issues
            learning_rate=0.001
        )
        
        # Save the model
        torch.save(detector.model.state_dict(), 'slum_detector.pth')
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 