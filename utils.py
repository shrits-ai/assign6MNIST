import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import logging
import os
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Set up logging
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 10)
        self.dropout = nn.Dropout(0.03)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

def get_data_loaders():
    # Training data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test data transforms (no augmentation needed)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transform)
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

class TrainingMetrics:
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        
    def update(self, train_loss, train_acc, test_loss, test_acc):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.test_losses.append(test_loss)
        self.test_accs.append(test_acc)
        
    def check_overfitting(self, patience=3):
        """
        Check for overfitting indicators
        Returns (is_overfitting, reasons) tuple
        """
        if len(self.train_losses) < patience + 1:
            return False, []  # Return tuple with empty list
            
        # Check if training accuracy is much higher than test accuracy
        acc_diff = self.train_accs[-1] - self.test_accs[-1]
        
        # Check if test loss is increasing while train loss is decreasing
        recent_train_loss_trend = self.train_losses[-1] - self.train_losses[-patience]
        recent_test_loss_trend = self.test_losses[-1] - self.test_losses[-patience]
        
        is_overfitting = False
        reasons = []
        
        # Accuracy gap check
        if acc_diff > 10:  # If training accuracy is 10% higher than test accuracy
            is_overfitting = True
            reasons.append(f"Large accuracy gap: Train acc is {acc_diff:.2f}% higher than test acc")
            
        # Loss trend check
        if recent_train_loss_trend < 0 and recent_test_loss_trend > 0:
            is_overfitting = True
            reasons.append("Train loss decreasing while test loss increasing")
            
        return is_overfitting, reasons  # Always return a tuple of (bool, list)