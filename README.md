# Project Name

[![ML Pipeline](https://img.shields.io/badge/ML%20Pipeline-Active-success)](https://github.com/shrits-ai/assign6MNIST/actions)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![GitHub issues](https://img.shields.io/github/issues/shrits-ai/assign6MNIST)](https://github.com/shrits-ai/assign6MNIST/issues)
[![Parameters](https://img.shields.io/badge/Total%20Parameters-11.5K-brightgreen)](https://github.com/shrits-ai/assign6MNIST)
[![BatchNorm](https://img.shields.io/badge/Batch%20Normalization-Yes-success)](https://github.com/shrits-ai/assign6MNIST)
[![Dropout](https://img.shields.io/badge/Dropout-0.1-informational)](https://github.com/shrits-ai/assign6MNIST)
[![Architecture](https://img.shields.io/badge/Final%20Layer-FC-yellow)](https://github.com/shrits-ai/assign6MNIST)

## Training Results

Target: Training stops at 99.40% test accuracy

### Best Results:
- Best Test Accuracy: 99.40% (Epoch 14)
- Training Time: 4.25 minutes
- Parameters: 11,466

### Model Architecture:
- Input Layer: MNIST images (1x28x28)
- Conv1: 8 channels, 3x3, padding=1
- BatchNorm1 + ReLU + MaxPool
- Dropout(0.1)
- Conv2: 16 channels, 3x3, padding=1
- BatchNorm2 + ReLU
- Conv3: 16 channels, 3x3, padding=1
- MaxPool
- Dropout(0.1)
- Fully Connected: 784 → 10

### Training Configuration:
- Optimizer: SGD with Nesterov Momentum
- Learning Rate: 0.05
- Batch Size: 64
- Weight Decay: 5e-4
- Early Stopping Patience: 5
- Max Epochs: 20

### Data Augmentation:
- Random Rotation (±10°)
- Random Translation (±10%)
- Normalization (mean=0.1307, std=0.3081)

### Results by Epoch:
```
Epoch 1:  Train Acc: 88.92% | Test Acc: 97.98% | Train Loss: 0.3468 | Test Loss: 0.0010
Epoch 2:  Train Acc: 95.50% | Test Acc: 98.52% | Train Loss: 0.1440 | Test Loss: 0.0007
Epoch 3:  Train Acc: 96.29% | Test Acc: 98.61% | Train Loss: 0.1208 | Test Loss: 0.0006
Epoch 4:  Train Acc: 96.75% | Test Acc: 98.88% | Train Loss: 0.1067 | Test Loss: 0.0005
Epoch 5:  Train Acc: 96.87% | Test Acc: 99.02% | Train Loss: 0.0986 | Test Loss: 0.0005
Epoch 6:  Train Acc: 96.85% | Test Acc: 98.96% | Train Loss: 0.0989 | Test Loss: 0.0005
Epoch 7:  Train Acc: 97.19% | Test Acc: 99.01% | Train Loss: 0.0897 | Test Loss: 0.0005
Epoch 8:  Train Acc: 97.17% | Test Acc: 99.04% | Train Loss: 0.0901 | Test Loss: 0.0005
Epoch 9:  Train Acc: 97.37% | Test Acc: 99.07% | Train Loss: 0.0852 | Test Loss: 0.0004
Epoch 10: Train Acc: 97.43% | Test Acc: 99.21% | Train Loss: 0.0836 | Test Loss: 0.0004
Epoch 11: Train Acc: 97.49% | Test Acc: 99.10% | Train Loss: 0.0814 | Test Loss: 0.0004
Epoch 12: Train Acc: 97.55% | Test Acc: 99.21% | Train Loss: 0.0772 | Test Loss: 0.0004
Epoch 13: Train Acc: 97.51% | Test Acc: 99.14% | Train Loss: 0.0796 | Test Loss: 0.0004
Epoch 14: Train Acc: 97.52% | Test Acc: 99.40% | Train Loss: 0.0786 | Test Loss: 0.0004
```

### Key Observations:
1. Reached target accuracy of 99.40% at epoch 14
2. No overfitting observed (train-test gap remains small)
3. Consistent improvement in test accuracy
4. Stable training with OneCycleLR scheduler
```
