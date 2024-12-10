# MNIST ERAV3 ASSIGNMENT 6

[![ML Pipeline](https://github.com/shrits-ai/assign6MNIST/actions/workflows/train.yml/badge.svg)](https://github.com/shrits-ai/assign6MNIST/actions/workflows/train.yml)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![GitHub issues](https://img.shields.io/github/issues/shrits-ai/assign6MNIST)](https://github.com/shrits-ai/assign6MNIST/issues)
[![Parameters](https://img.shields.io/badge/Total%20Parameters-11.5K-brightgreen)](https://github.com/shrits-ai/assign6MNIST)
[![BatchNorm](https://img.shields.io/badge/Batch%20Normalization-Yes-success)](https://github.com/shrits-ai/assign6MNIST)
[![Dropout](https://img.shields.io/badge/Dropout-0.1-informational)](https://github.com/shrits-ai/assign6MNIST)
[![Architecture](https://img.shields.io/badge/Final%20Layer-FC-yellow)](https://github.com/shrits-ai/assign6MNIST)

## Training Results

### Best Results:
- Best Test Accuracy: 99.40% (Epoch 14)
- Training Time: 5.77 minutes
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

### Test Cases:
- Layer Dimension Tests:
  - Checks output shape of each layer
  - Verifies correct dimensionality through the network
- Output Properties Tests:
  - Verifies softmax probabilities sum to 1
  - Checks output shape for batched input
- Parameter Tests:
  - Verifies total parameter count < 20K
  - Checks BatchNorm parameters
- Training Mode Tests:
  - Verifies dropout behavior differs between train/eval modes
  - Tests forward/backward pass
- Model Training Test:
  - Tests a complete training step
  - Verifies optimizer and loss function work
- Batch Processing:
  - Tests model with batch input
  - Verifies batch dimension handling
- Architectural Constraints:
  - Checks for required layers
  - Verifies dropout rate range
  - Tests BatchNorm configuration

### Results by Epoch (Formatted):
```
2024-12-10 16:30:27,930 - Training Epoch 1: Average loss: 0.3468, Accuracy: 88.92%
2024-12-10 16:30:28,752 - Test set: Average loss: 0.0010, Accuracy: 9798/10000 (97.98%)
2024-12-10 16:30:45,682 - Training Epoch 2: Average loss: 0.1440, Accuracy: 95.50%
2024-12-10 16:30:46,508 - Test set: Average loss: 0.0007, Accuracy: 9852/10000 (98.52%)
2024-12-10 16:31:05,854 - Training Epoch 3: Average loss: 0.1208, Accuracy: 96.29%
2024-12-10 16:31:06,848 - Test set: Average loss: 0.0006, Accuracy: 9861/10000 (98.61%)
2024-12-10 16:31:26,552 - Training Epoch 4: Average loss: 0.1067, Accuracy: 96.75%
2024-12-10 16:31:27,502 - Test set: Average loss: 0.0005, Accuracy: 9888/10000 (98.88%)
2024-12-10 16:31:47,186 - Training Epoch 5: Average loss: 0.0986, Accuracy: 96.87%
2024-12-10 16:31:48,166 - Test set: Average loss: 0.0005, Accuracy: 9902/10000 (99.02%)
2024-12-10 16:32:05,569 - Training Epoch 6: Average loss: 0.0989, Accuracy: 96.85%
2024-12-10 16:32:06,436 - Test set: Average loss: 0.0005, Accuracy: 9896/10000 (98.96%)
2024-12-10 16:32:23,325 - Training Epoch 7: Average loss: 0.0897, Accuracy: 97.19%
2024-12-10 16:32:24,192 - Test set: Average loss: 0.0005, Accuracy: 9901/10000 (99.01%)
2024-12-10 16:32:41,112 - Training Epoch 8: Average loss: 0.0901, Accuracy: 97.17%
2024-12-10 16:32:41,922 - Test set: Average loss: 0.0005, Accuracy: 9904/10000 (99.04%)
2024-12-10 16:32:58,860 - Training Epoch 9: Average loss: 0.0852, Accuracy: 97.37%
2024-12-10 16:32:59,662 - Test set: Average loss: 0.0004, Accuracy: 9907/10000 (99.07%)
2024-12-10 16:33:16,560 - Training Epoch 10: Average loss: 0.0836, Accuracy: 97.43%
2024-12-10 16:33:17,389 - Test set: Average loss: 0.0004, Accuracy: 9921/10000 (99.21%)
2024-12-10 16:33:34,164 - Training Epoch 11: Average loss: 0.0814, Accuracy: 97.49%
2024-12-10 16:33:35,000 - Test set: Average loss: 0.0004, Accuracy: 9910/10000 (99.10%)
2024-12-10 16:33:51,877 - Training Epoch 12: Average loss: 0.0772, Accuracy: 97.55%
2024-12-10 16:33:52,765 - Test set: Average loss: 0.0004, Accuracy: 9921/10000 (99.21%)
2024-12-10 16:34:09,776 - Training Epoch 13: Average loss: 0.0796, Accuracy: 97.51%
2024-12-10 16:34:10,635 - Test set: Average loss: 0.0004, Accuracy: 9914/10000 (99.14%)
2024-12-10 16:34:27,425 - Training Epoch 14: Average loss: 0.0786, Accuracy: 97.52%
2024-12-10 16:34:28,272 - Test set: Average loss: 0.0004, Accuracy: 9940/10000 (99.40%)
```
### Key Observations:
1. Best accuracy of 99.40% achieved at epoch 14
2. No significant overfitting (train-test gap remains small)
3. Consistent improvement in test accuracy
4. Stable training with OneCycleLR scheduler
5. Early stopping triggered after epoch 19

### Actual Training logs: 
```
(venv) shriti@Shritis-MacBook-Pro fresh_repo % python3 train.py 
2024-12-10 16:30:10,880 - Using device: cpu
2024-12-10 16:30:10,881 - Python version: 3.11.6 (v3.11.6:8b6ee5ba3b, Oct  2 2023, 11:18:21) [Clang 13.0.0 (clang-1300.0.29.30)]
2024-12-10 16:30:10,881 - PyTorch version: 2.5.1
2024-12-10 16:30:10,881 - Loading datasets...
2024-12-10 16:30:10,929 - Training samples: 60000
2024-12-10 16:30:10,929 - Test samples: 10000
2024-12-10 16:30:10,929 - Initializing model...
2024-12-10 16:30:10,938 - 
Model Architecture:
Net(
  (conv1): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=784, out_features=10, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
)
2024-12-10 16:30:10,938 - Total parameters: 11,466
2024-12-10 16:30:10,938 - 
Starting training...
2024-12-10 16:30:10,938 - 
Epoch 1/20
2024-12-10 16:30:11,024 - Train Epoch: 1 [0/60000 (0%)] Loss: 2.261631
2024-12-10 16:30:12,848 - Train Epoch: 1 [6400/60000 (11%)]     Loss: 0.585811
2024-12-10 16:30:14,638 - Train Epoch: 1 [12800/60000 (21%)]    Loss: 0.420120
2024-12-10 16:30:16,464 - Train Epoch: 1 [19200/60000 (32%)]    Loss: 0.166534
2024-12-10 16:30:18,304 - Train Epoch: 1 [25600/60000 (43%)]    Loss: 0.145863
2024-12-10 16:30:20,051 - Train Epoch: 1 [32000/60000 (53%)]    Loss: 0.205739
2024-12-10 16:30:21,924 - Train Epoch: 1 [38400/60000 (64%)]    Loss: 0.199692
2024-12-10 16:30:23,725 - Train Epoch: 1 [44800/60000 (75%)]    Loss: 0.275986
2024-12-10 16:30:25,518 - Train Epoch: 1 [51200/60000 (85%)]    Loss: 0.205317
2024-12-10 16:30:27,279 - Train Epoch: 1 [57600/60000 (96%)]    Loss: 0.121506
2024-12-10 16:30:27,930 - Training Epoch 1: Average loss: 0.3468, Accuracy: 88.92%
2024-12-10 16:30:28,752 - Test set: Average loss: 0.0010, Accuracy: 9798/10000 (97.98%)
2024-12-10 16:30:28,752 - Epoch 1 completed in 17.81s
2024-12-10 16:30:28,752 - Train Loss: 0.3468, Train Acc: 88.92%
2024-12-10 16:30:28,752 - Test Loss: 0.0010, Test Acc: 97.98%
2024-12-10 16:30:28,752 - Train/Test Accuracy Gap: -9.06%
2024-12-10 16:30:28,755 - New best accuracy: 97.98% at epoch 1
2024-12-10 16:30:28,755 - 
Epoch 2/20
2024-12-10 16:30:28,778 - Train Epoch: 2 [0/60000 (0%)] Loss: 0.176540
2024-12-10 16:30:30,608 - Train Epoch: 2 [6400/60000 (11%)]     Loss: 0.242801
2024-12-10 16:30:32,398 - Train Epoch: 2 [12800/60000 (21%)]    Loss: 0.114498
2024-12-10 16:30:34,170 - Train Epoch: 2 [19200/60000 (32%)]    Loss: 0.090069
2024-12-10 16:30:36,063 - Train Epoch: 2 [25600/60000 (43%)]    Loss: 0.096316
2024-12-10 16:30:37,836 - Train Epoch: 2 [32000/60000 (53%)]    Loss: 0.155359
2024-12-10 16:30:39,606 - Train Epoch: 2 [38400/60000 (64%)]    Loss: 0.119381
2024-12-10 16:30:41,423 - Train Epoch: 2 [44800/60000 (75%)]    Loss: 0.116294
2024-12-10 16:30:43,207 - Train Epoch: 2 [51200/60000 (85%)]    Loss: 0.151498
2024-12-10 16:30:44,991 - Train Epoch: 2 [57600/60000 (96%)]    Loss: 0.201881
2024-12-10 16:30:45,682 - Training Epoch 2: Average loss: 0.1440, Accuracy: 95.50%
2024-12-10 16:30:46,508 - Test set: Average loss: 0.0007, Accuracy: 9852/10000 (98.52%)
2024-12-10 16:30:46,508 - Epoch 2 completed in 17.75s
2024-12-10 16:30:46,508 - Train Loss: 0.1440, Train Acc: 95.50%
2024-12-10 16:30:46,508 - Test Loss: 0.0007, Test Acc: 98.52%
2024-12-10 16:30:46,508 - Train/Test Accuracy Gap: -3.02%
2024-12-10 16:30:46,509 - New best accuracy: 98.52% at epoch 2
2024-12-10 16:30:46,509 - 
Epoch 3/20
2024-12-10 16:30:46,529 - Train Epoch: 3 [0/60000 (0%)] Loss: 0.208899
2024-12-10 16:30:48,348 - Train Epoch: 3 [6400/60000 (11%)]     Loss: 0.081802
2024-12-10 16:30:50,176 - Train Epoch: 3 [12800/60000 (21%)]    Loss: 0.210345
2024-12-10 16:30:52,264 - Train Epoch: 3 [19200/60000 (32%)]    Loss: 0.090674
2024-12-10 16:30:54,360 - Train Epoch: 3 [25600/60000 (43%)]    Loss: 0.182162
2024-12-10 16:30:56,568 - Train Epoch: 3 [32000/60000 (53%)]    Loss: 0.101445
2024-12-10 16:30:58,696 - Train Epoch: 3 [38400/60000 (64%)]    Loss: 0.289800
2024-12-10 16:31:00,837 - Train Epoch: 3 [44800/60000 (75%)]    Loss: 0.174301
2024-12-10 16:31:02,905 - Train Epoch: 3 [51200/60000 (85%)]    Loss: 0.190923
2024-12-10 16:31:04,964 - Train Epoch: 3 [57600/60000 (96%)]    Loss: 0.149641
2024-12-10 16:31:05,854 - Training Epoch 3: Average loss: 0.1208, Accuracy: 96.29%
2024-12-10 16:31:06,848 - Test set: Average loss: 0.0006, Accuracy: 9861/10000 (98.61%)
2024-12-10 16:31:06,848 - Epoch 3 completed in 20.34s
2024-12-10 16:31:06,848 - Train Loss: 0.1208, Train Acc: 96.29%
2024-12-10 16:31:06,848 - Test Loss: 0.0006, Test Acc: 98.61%
2024-12-10 16:31:06,848 - Train/Test Accuracy Gap: -2.32%
2024-12-10 16:31:06,849 - New best accuracy: 98.61% at epoch 3
2024-12-10 16:31:06,849 - 
Epoch 4/20
2024-12-10 16:31:06,875 - Train Epoch: 4 [0/60000 (0%)] Loss: 0.165462
2024-12-10 16:31:08,957 - Train Epoch: 4 [6400/60000 (11%)]     Loss: 0.019137
2024-12-10 16:31:11,095 - Train Epoch: 4 [12800/60000 (21%)]    Loss: 0.147641
2024-12-10 16:31:13,175 - Train Epoch: 4 [19200/60000 (32%)]    Loss: 0.073307
2024-12-10 16:31:15,225 - Train Epoch: 4 [25600/60000 (43%)]    Loss: 0.170970
2024-12-10 16:31:17,315 - Train Epoch: 4 [32000/60000 (53%)]    Loss: 0.059345
2024-12-10 16:31:19,435 - Train Epoch: 4 [38400/60000 (64%)]    Loss: 0.222884
2024-12-10 16:31:21,529 - Train Epoch: 4 [44800/60000 (75%)]    Loss: 0.032678
2024-12-10 16:31:23,631 - Train Epoch: 4 [51200/60000 (85%)]    Loss: 0.006545
2024-12-10 16:31:25,765 - Train Epoch: 4 [57600/60000 (96%)]    Loss: 0.124323
2024-12-10 16:31:26,552 - Training Epoch 4: Average loss: 0.1067, Accuracy: 96.75%
2024-12-10 16:31:27,502 - Test set: Average loss: 0.0005, Accuracy: 9888/10000 (98.88%)
2024-12-10 16:31:27,502 - Epoch 4 completed in 20.65s
2024-12-10 16:31:27,502 - Train Loss: 0.1067, Train Acc: 96.75%
2024-12-10 16:31:27,502 - Test Loss: 0.0005, Test Acc: 98.88%
2024-12-10 16:31:27,502 - Train/Test Accuracy Gap: -2.13%
2024-12-10 16:31:27,503 - New best accuracy: 98.88% at epoch 4
2024-12-10 16:31:27,503 - 
Epoch 5/20
2024-12-10 16:31:27,526 - Train Epoch: 5 [0/60000 (0%)] Loss: 0.165319
2024-12-10 16:31:29,661 - Train Epoch: 5 [6400/60000 (11%)]     Loss: 0.083702
2024-12-10 16:31:31,823 - Train Epoch: 5 [12800/60000 (21%)]    Loss: 0.073090
2024-12-10 16:31:33,815 - Train Epoch: 5 [19200/60000 (32%)]    Loss: 0.123575
2024-12-10 16:31:35,913 - Train Epoch: 5 [25600/60000 (43%)]    Loss: 0.081372
2024-12-10 16:31:38,042 - Train Epoch: 5 [32000/60000 (53%)]    Loss: 0.094283
2024-12-10 16:31:40,076 - Train Epoch: 5 [38400/60000 (64%)]    Loss: 0.125371
2024-12-10 16:31:42,236 - Train Epoch: 5 [44800/60000 (75%)]    Loss: 0.062229
2024-12-10 16:31:44,296 - Train Epoch: 5 [51200/60000 (85%)]    Loss: 0.181835
2024-12-10 16:31:46,419 - Train Epoch: 5 [57600/60000 (96%)]    Loss: 0.054964
2024-12-10 16:31:47,186 - Training Epoch 5: Average loss: 0.0986, Accuracy: 96.87%
2024-12-10 16:31:48,166 - Test set: Average loss: 0.0005, Accuracy: 9902/10000 (99.02%)
2024-12-10 16:31:48,166 - Epoch 5 completed in 20.66s
2024-12-10 16:31:48,166 - Train Loss: 0.0986, Train Acc: 96.87%
2024-12-10 16:31:48,166 - Test Loss: 0.0005, Test Acc: 99.02%
2024-12-10 16:31:48,166 - Train/Test Accuracy Gap: -2.15%
2024-12-10 16:31:48,168 - New best accuracy: 99.02% at epoch 5
2024-12-10 16:31:48,168 - 
Epoch 6/20
2024-12-10 16:31:48,191 - Train Epoch: 6 [0/60000 (0%)] Loss: 0.085614
2024-12-10 16:31:50,340 - Train Epoch: 6 [6400/60000 (11%)]     Loss: 0.085339
2024-12-10 16:31:52,346 - Train Epoch: 6 [12800/60000 (21%)]    Loss: 0.206654
2024-12-10 16:31:54,114 - Train Epoch: 6 [19200/60000 (32%)]    Loss: 0.119421
2024-12-10 16:31:55,963 - Train Epoch: 6 [25600/60000 (43%)]    Loss: 0.176617
2024-12-10 16:31:57,792 - Train Epoch: 6 [32000/60000 (53%)]    Loss: 0.071272
2024-12-10 16:31:59,590 - Train Epoch: 6 [38400/60000 (64%)]    Loss: 0.051543
2024-12-10 16:32:01,361 - Train Epoch: 6 [44800/60000 (75%)]    Loss: 0.020817
2024-12-10 16:32:03,129 - Train Epoch: 6 [51200/60000 (85%)]    Loss: 0.099820
2024-12-10 16:32:04,855 - Train Epoch: 6 [57600/60000 (96%)]    Loss: 0.035642
2024-12-10 16:32:05,569 - Training Epoch 6: Average loss: 0.0989, Accuracy: 96.85%
2024-12-10 16:32:06,436 - Test set: Average loss: 0.0005, Accuracy: 9896/10000 (98.96%)
2024-12-10 16:32:06,436 - Epoch 6 completed in 18.27s
2024-12-10 16:32:06,436 - Train Loss: 0.0989, Train Acc: 96.85%
2024-12-10 16:32:06,436 - Test Loss: 0.0005, Test Acc: 98.96%
2024-12-10 16:32:06,436 - Train/Test Accuracy Gap: -2.11%
2024-12-10 16:32:06,436 - 
Epoch 7/20
2024-12-10 16:32:06,456 - Train Epoch: 7 [0/60000 (0%)] Loss: 0.101909
2024-12-10 16:32:08,205 - Train Epoch: 7 [6400/60000 (11%)]     Loss: 0.069237
2024-12-10 16:32:09,970 - Train Epoch: 7 [12800/60000 (21%)]    Loss: 0.059441
2024-12-10 16:32:11,977 - Train Epoch: 7 [19200/60000 (32%)]    Loss: 0.021996
2024-12-10 16:32:13,725 - Train Epoch: 7 [25600/60000 (43%)]    Loss: 0.030625
2024-12-10 16:32:15,522 - Train Epoch: 7 [32000/60000 (53%)]    Loss: 0.023674
2024-12-10 16:32:17,256 - Train Epoch: 7 [38400/60000 (64%)]    Loss: 0.060946
2024-12-10 16:32:19,031 - Train Epoch: 7 [44800/60000 (75%)]    Loss: 0.010261
2024-12-10 16:32:20,862 - Train Epoch: 7 [51200/60000 (85%)]    Loss: 0.126444
2024-12-10 16:32:22,659 - Train Epoch: 7 [57600/60000 (96%)]    Loss: 0.063692
2024-12-10 16:32:23,325 - Training Epoch 7: Average loss: 0.0897, Accuracy: 97.19%
2024-12-10 16:32:24,192 - Test set: Average loss: 0.0005, Accuracy: 9901/10000 (99.01%)
2024-12-10 16:32:24,192 - Epoch 7 completed in 17.76s
2024-12-10 16:32:24,192 - Train Loss: 0.0897, Train Acc: 97.19%
2024-12-10 16:32:24,192 - Test Loss: 0.0005, Test Acc: 99.01%
2024-12-10 16:32:24,192 - Train/Test Accuracy Gap: -1.83%
2024-12-10 16:32:24,192 - 
Epoch 8/20
2024-12-10 16:32:24,213 - Train Epoch: 8 [0/60000 (0%)] Loss: 0.043772
2024-12-10 16:32:26,106 - Train Epoch: 8 [6400/60000 (11%)]     Loss: 0.049751
2024-12-10 16:32:27,857 - Train Epoch: 8 [12800/60000 (21%)]    Loss: 0.065558
2024-12-10 16:32:29,608 - Train Epoch: 8 [19200/60000 (32%)]    Loss: 0.037603
2024-12-10 16:32:31,425 - Train Epoch: 8 [25600/60000 (43%)]    Loss: 0.124857
2024-12-10 16:32:33,205 - Train Epoch: 8 [32000/60000 (53%)]    Loss: 0.136880
2024-12-10 16:32:34,975 - Train Epoch: 8 [38400/60000 (64%)]    Loss: 0.041854
2024-12-10 16:32:36,821 - Train Epoch: 8 [44800/60000 (75%)]    Loss: 0.053387
2024-12-10 16:32:38,601 - Train Epoch: 8 [51200/60000 (85%)]    Loss: 0.079373
2024-12-10 16:32:40,475 - Train Epoch: 8 [57600/60000 (96%)]    Loss: 0.039419
2024-12-10 16:32:41,112 - Training Epoch 8: Average loss: 0.0901, Accuracy: 97.17%
2024-12-10 16:32:41,922 - Test set: Average loss: 0.0005, Accuracy: 9904/10000 (99.04%)
2024-12-10 16:32:41,923 - Epoch 8 completed in 17.73s
2024-12-10 16:32:41,923 - Train Loss: 0.0901, Train Acc: 97.17%
2024-12-10 16:32:41,923 - Test Loss: 0.0005, Test Acc: 99.04%
2024-12-10 16:32:41,923 - Train/Test Accuracy Gap: -1.87%
2024-12-10 16:32:41,924 - New best accuracy: 99.04% at epoch 8
2024-12-10 16:32:41,924 - 
Epoch 9/20
2024-12-10 16:32:41,944 - Train Epoch: 9 [0/60000 (0%)] Loss: 0.130585
2024-12-10 16:32:43,802 - Train Epoch: 9 [6400/60000 (11%)]     Loss: 0.073250
2024-12-10 16:32:45,612 - Train Epoch: 9 [12800/60000 (21%)]    Loss: 0.206532
2024-12-10 16:32:47,356 - Train Epoch: 9 [19200/60000 (32%)]    Loss: 0.108508
2024-12-10 16:32:49,111 - Train Epoch: 9 [25600/60000 (43%)]    Loss: 0.059363
2024-12-10 16:32:50,943 - Train Epoch: 9 [32000/60000 (53%)]    Loss: 0.019042
2024-12-10 16:32:52,735 - Train Epoch: 9 [38400/60000 (64%)]    Loss: 0.062867
2024-12-10 16:32:54,477 - Train Epoch: 9 [44800/60000 (75%)]    Loss: 0.089432
2024-12-10 16:32:56,393 - Train Epoch: 9 [51200/60000 (85%)]    Loss: 0.140686
2024-12-10 16:32:58,207 - Train Epoch: 9 [57600/60000 (96%)]    Loss: 0.124031
2024-12-10 16:32:58,860 - Training Epoch 9: Average loss: 0.0852, Accuracy: 97.37%
2024-12-10 16:32:59,662 - Test set: Average loss: 0.0004, Accuracy: 9907/10000 (99.07%)
2024-12-10 16:32:59,662 - Epoch 9 completed in 17.74s
2024-12-10 16:32:59,662 - Train Loss: 0.0852, Train Acc: 97.37%
2024-12-10 16:32:59,662 - Test Loss: 0.0004, Test Acc: 99.07%
2024-12-10 16:32:59,662 - Train/Test Accuracy Gap: -1.70%
2024-12-10 16:32:59,663 - New best accuracy: 99.07% at epoch 9
2024-12-10 16:32:59,663 - 
Epoch 10/20
2024-12-10 16:32:59,684 - Train Epoch: 10 [0/60000 (0%)]        Loss: 0.021904
2024-12-10 16:33:01,503 - Train Epoch: 10 [6400/60000 (11%)]    Loss: 0.167866
2024-12-10 16:33:03,308 - Train Epoch: 10 [12800/60000 (21%)]   Loss: 0.108206
2024-12-10 16:33:05,068 - Train Epoch: 10 [19200/60000 (32%)]   Loss: 0.111826
2024-12-10 16:33:06,962 - Train Epoch: 10 [25600/60000 (43%)]   Loss: 0.083317
2024-12-10 16:33:08,718 - Train Epoch: 10 [32000/60000 (53%)]   Loss: 0.038512
2024-12-10 16:33:10,540 - Train Epoch: 10 [38400/60000 (64%)]   Loss: 0.037575
2024-12-10 16:33:12,323 - Train Epoch: 10 [44800/60000 (75%)]   Loss: 0.069276
2024-12-10 16:33:14,090 - Train Epoch: 10 [51200/60000 (85%)]   Loss: 0.052251
2024-12-10 16:33:15,907 - Train Epoch: 10 [57600/60000 (96%)]   Loss: 0.138949
2024-12-10 16:33:16,560 - Training Epoch 10: Average loss: 0.0836, Accuracy: 97.43%
2024-12-10 16:33:17,389 - Test set: Average loss: 0.0004, Accuracy: 9921/10000 (99.21%)
2024-12-10 16:33:17,390 - Epoch 10 completed in 17.73s
2024-12-10 16:33:17,390 - Train Loss: 0.0836, Train Acc: 97.43%
2024-12-10 16:33:17,390 - Test Loss: 0.0004, Test Acc: 99.21%
2024-12-10 16:33:17,390 - Train/Test Accuracy Gap: -1.78%
2024-12-10 16:33:17,391 - New best accuracy: 99.21% at epoch 10
2024-12-10 16:33:17,391 - 
Epoch 11/20
2024-12-10 16:33:17,412 - Train Epoch: 11 [0/60000 (0%)]        Loss: 0.058170
2024-12-10 16:33:19,196 - Train Epoch: 11 [6400/60000 (11%)]    Loss: 0.078159
2024-12-10 16:33:21,011 - Train Epoch: 11 [12800/60000 (21%)]   Loss: 0.270729
2024-12-10 16:33:22,872 - Train Epoch: 11 [19200/60000 (32%)]   Loss: 0.021814
2024-12-10 16:33:24,637 - Train Epoch: 11 [25600/60000 (43%)]   Loss: 0.067750
2024-12-10 16:33:26,433 - Train Epoch: 11 [32000/60000 (53%)]   Loss: 0.093830
2024-12-10 16:33:28,184 - Train Epoch: 11 [38400/60000 (64%)]   Loss: 0.179415
2024-12-10 16:33:29,944 - Train Epoch: 11 [44800/60000 (75%)]   Loss: 0.045116
2024-12-10 16:33:31,743 - Train Epoch: 11 [51200/60000 (85%)]   Loss: 0.043615
2024-12-10 16:33:33,513 - Train Epoch: 11 [57600/60000 (96%)]   Loss: 0.051887
2024-12-10 16:33:34,164 - Training Epoch 11: Average loss: 0.0814, Accuracy: 97.49%
2024-12-10 16:33:35,000 - Test set: Average loss: 0.0004, Accuracy: 9910/10000 (99.10%)
2024-12-10 16:33:35,001 - Epoch 11 completed in 17.61s
2024-12-10 16:33:35,001 - Train Loss: 0.0814, Train Acc: 97.49%
2024-12-10 16:33:35,001 - Test Loss: 0.0004, Test Acc: 99.10%
2024-12-10 16:33:35,001 - Train/Test Accuracy Gap: -1.61%
2024-12-10 16:33:35,001 - 
Epoch 12/20
2024-12-10 16:33:35,029 - Train Epoch: 12 [0/60000 (0%)]        Loss: 0.030316
2024-12-10 16:33:36,894 - Train Epoch: 12 [6400/60000 (11%)]    Loss: 0.126528
2024-12-10 16:33:38,704 - Train Epoch: 12 [12800/60000 (21%)]   Loss: 0.047872
2024-12-10 16:33:40,499 - Train Epoch: 12 [19200/60000 (32%)]   Loss: 0.013330
2024-12-10 16:33:42,266 - Train Epoch: 12 [25600/60000 (43%)]   Loss: 0.143308
2024-12-10 16:33:44,038 - Train Epoch: 12 [32000/60000 (53%)]   Loss: 0.048538
2024-12-10 16:33:45,847 - Train Epoch: 12 [38400/60000 (64%)]   Loss: 0.034506
2024-12-10 16:33:47,619 - Train Epoch: 12 [44800/60000 (75%)]   Loss: 0.017903
2024-12-10 16:33:49,371 - Train Epoch: 12 [51200/60000 (85%)]   Loss: 0.047664
2024-12-10 16:33:51,239 - Train Epoch: 12 [57600/60000 (96%)]   Loss: 0.060986
2024-12-10 16:33:51,877 - Training Epoch 12: Average loss: 0.0772, Accuracy: 97.55%
2024-12-10 16:33:52,765 - Test set: Average loss: 0.0004, Accuracy: 9921/10000 (99.21%)
2024-12-10 16:33:52,765 - Epoch 12 completed in 17.76s
2024-12-10 16:33:52,765 - Train Loss: 0.0772, Train Acc: 97.55%
2024-12-10 16:33:52,765 - Test Loss: 0.0004, Test Acc: 99.21%
2024-12-10 16:33:52,765 - Train/Test Accuracy Gap: -1.66%
2024-12-10 16:33:52,765 - 
Epoch 13/20
2024-12-10 16:33:52,787 - Train Epoch: 13 [0/60000 (0%)]        Loss: 0.039936
2024-12-10 16:33:54,618 - Train Epoch: 13 [6400/60000 (11%)]    Loss: 0.026704
2024-12-10 16:33:56,420 - Train Epoch: 13 [12800/60000 (21%)]   Loss: 0.045928
2024-12-10 16:33:58,238 - Train Epoch: 13 [19200/60000 (32%)]   Loss: 0.059882
2024-12-10 16:33:59,981 - Train Epoch: 13 [25600/60000 (43%)]   Loss: 0.058368
2024-12-10 16:34:01,773 - Train Epoch: 13 [32000/60000 (53%)]   Loss: 0.089992
2024-12-10 16:34:03,608 - Train Epoch: 13 [38400/60000 (64%)]   Loss: 0.122840
2024-12-10 16:34:05,509 - Train Epoch: 13 [44800/60000 (75%)]   Loss: 0.012978
2024-12-10 16:34:07,364 - Train Epoch: 13 [51200/60000 (85%)]   Loss: 0.188200
2024-12-10 16:34:09,143 - Train Epoch: 13 [57600/60000 (96%)]   Loss: 0.033810
2024-12-10 16:34:09,776 - Training Epoch 13: Average loss: 0.0796, Accuracy: 97.51%
2024-12-10 16:34:10,635 - Test set: Average loss: 0.0004, Accuracy: 9914/10000 (99.14%)
2024-12-10 16:34:10,635 - Epoch 13 completed in 17.87s
2024-12-10 16:34:10,635 - Train Loss: 0.0796, Train Acc: 97.51%
2024-12-10 16:34:10,635 - Test Loss: 0.0004, Test Acc: 99.14%
2024-12-10 16:34:10,635 - Train/Test Accuracy Gap: -1.63%
2024-12-10 16:34:10,635 - 
Epoch 14/20
2024-12-10 16:34:10,672 - Train Epoch: 14 [0/60000 (0%)]        Loss: 0.154248
2024-12-10 16:34:12,446 - Train Epoch: 14 [6400/60000 (11%)]    Loss: 0.142316
2024-12-10 16:34:14,219 - Train Epoch: 14 [12800/60000 (21%)]   Loss: 0.021765
2024-12-10 16:34:16,026 - Train Epoch: 14 [19200/60000 (32%)]   Loss: 0.028119
2024-12-10 16:34:17,778 - Train Epoch: 14 [25600/60000 (43%)]   Loss: 0.071708
2024-12-10 16:34:19,578 - Train Epoch: 14 [32000/60000 (53%)]   Loss: 0.026756
2024-12-10 16:34:21,430 - Train Epoch: 14 [38400/60000 (64%)]   Loss: 0.043671
2024-12-10 16:34:23,193 - Train Epoch: 14 [44800/60000 (75%)]   Loss: 0.078260
2024-12-10 16:34:24,959 - Train Epoch: 14 [51200/60000 (85%)]   Loss: 0.045793
2024-12-10 16:34:26,779 - Train Epoch: 14 [57600/60000 (96%)]   Loss: 0.186354
2024-12-10 16:34:27,425 - Training Epoch 14: Average loss: 0.0786, Accuracy: 97.52%
2024-12-10 16:34:28,272 - Test set: Average loss: 0.0004, Accuracy: 9940/10000 (99.40%)
2024-12-10 16:34:28,272 - Epoch 14 completed in 17.64s
2024-12-10 16:34:28,272 - Train Loss: 0.0786, Train Acc: 97.52%
2024-12-10 16:34:28,272 - Test Loss: 0.0004, Test Acc: 99.40%
2024-12-10 16:34:28,273 - Train/Test Accuracy Gap: -1.89%
2024-12-10 16:34:28,274 - New best accuracy: 99.40% at epoch 14
2024-12-10 16:34:28,274 - 
Epoch 15/20
2024-12-10 16:34:28,299 - Train Epoch: 15 [0/60000 (0%)]        Loss: 0.030875
2024-12-10 16:34:30,099 - Train Epoch: 15 [6400/60000 (11%)]    Loss: 0.142651
2024-12-10 16:34:31,893 - Train Epoch: 15 [12800/60000 (21%)]   Loss: 0.074071
2024-12-10 16:34:33,650 - Train Epoch: 15 [19200/60000 (32%)]   Loss: 0.200146
2024-12-10 16:34:35,400 - Train Epoch: 15 [25600/60000 (43%)]   Loss: 0.179926
2024-12-10 16:34:37,244 - Train Epoch: 15 [32000/60000 (53%)]   Loss: 0.015594
2024-12-10 16:34:38,980 - Train Epoch: 15 [38400/60000 (64%)]   Loss: 0.049600
2024-12-10 16:34:40,794 - Train Epoch: 15 [44800/60000 (75%)]   Loss: 0.021579
2024-12-10 16:34:42,539 - Train Epoch: 15 [51200/60000 (85%)]   Loss: 0.149052
2024-12-10 16:34:44,339 - Train Epoch: 15 [57600/60000 (96%)]   Loss: 0.062227
2024-12-10 16:34:44,997 - Training Epoch 15: Average loss: 0.0772, Accuracy: 97.62%
2024-12-10 16:34:45,848 - Test set: Average loss: 0.0004, Accuracy: 9919/10000 (99.19%)
2024-12-10 16:34:45,848 - Epoch 15 completed in 17.57s
2024-12-10 16:34:45,848 - Train Loss: 0.0772, Train Acc: 97.62%
2024-12-10 16:34:45,848 - Test Loss: 0.0004, Test Acc: 99.19%
2024-12-10 16:34:45,848 - Train/Test Accuracy Gap: -1.56%
2024-12-10 16:34:45,848 - 
Epoch 16/20
2024-12-10 16:34:45,870 - Train Epoch: 16 [0/60000 (0%)]        Loss: 0.044923
2024-12-10 16:34:47,622 - Train Epoch: 16 [6400/60000 (11%)]    Loss: 0.020166
2024-12-10 16:34:49,370 - Train Epoch: 16 [12800/60000 (21%)]   Loss: 0.024027
2024-12-10 16:34:51,206 - Train Epoch: 16 [19200/60000 (32%)]   Loss: 0.101921
2024-12-10 16:34:52,979 - Train Epoch: 16 [25600/60000 (43%)]   Loss: 0.103944
2024-12-10 16:34:54,778 - Train Epoch: 16 [32000/60000 (53%)]   Loss: 0.163257
2024-12-10 16:34:56,583 - Train Epoch: 16 [38400/60000 (64%)]   Loss: 0.058841
2024-12-10 16:34:58,347 - Train Epoch: 16 [44800/60000 (75%)]   Loss: 0.097670
2024-12-10 16:35:00,158 - Train Epoch: 16 [51200/60000 (85%)]   Loss: 0.025114
2024-12-10 16:35:01,951 - Train Epoch: 16 [57600/60000 (96%)]   Loss: 0.102670
2024-12-10 16:35:02,586 - Training Epoch 16: Average loss: 0.0748, Accuracy: 97.60%
2024-12-10 16:35:03,404 - Test set: Average loss: 0.0004, Accuracy: 9909/10000 (99.09%)
2024-12-10 16:35:03,404 - 
Possible overfitting detected!
2024-12-10 16:35:03,404 - - Train loss decreasing while test loss increasing
2024-12-10 16:35:03,404 - Consider:
1. Increasing dropout rate
2. Adding data augmentation
3. Reducing model complexity
4. Early stopping
2024-12-10 16:35:03,404 - Epoch 16 completed in 17.56s
2024-12-10 16:35:03,404 - Train Loss: 0.0748, Train Acc: 97.60%
2024-12-10 16:35:03,404 - Test Loss: 0.0004, Test Acc: 99.09%
2024-12-10 16:35:03,404 - Train/Test Accuracy Gap: -1.49%
2024-12-10 16:35:03,404 - 
Epoch 17/20
2024-12-10 16:35:03,431 - Train Epoch: 17 [0/60000 (0%)]        Loss: 0.096399
2024-12-10 16:35:05,243 - Train Epoch: 17 [6400/60000 (11%)]    Loss: 0.018645
2024-12-10 16:35:07,125 - Train Epoch: 17 [12800/60000 (21%)]   Loss: 0.144412
2024-12-10 16:35:09,027 - Train Epoch: 17 [19200/60000 (32%)]   Loss: 0.080013
2024-12-10 16:35:10,839 - Train Epoch: 17 [25600/60000 (43%)]   Loss: 0.040673
2024-12-10 16:35:12,811 - Train Epoch: 17 [32000/60000 (53%)]   Loss: 0.120355
2024-12-10 16:35:14,629 - Train Epoch: 17 [38400/60000 (64%)]   Loss: 0.061326
2024-12-10 16:35:16,487 - Train Epoch: 17 [44800/60000 (75%)]   Loss: 0.104589
2024-12-10 16:35:18,305 - Train Epoch: 17 [51200/60000 (85%)]   Loss: 0.091276
2024-12-10 16:35:20,185 - Train Epoch: 17 [57600/60000 (96%)]   Loss: 0.056930
2024-12-10 16:35:20,945 - Training Epoch 17: Average loss: 0.0741, Accuracy: 97.71%
2024-12-10 16:35:21,834 - Test set: Average loss: 0.0003, Accuracy: 9932/10000 (99.32%)
2024-12-10 16:35:21,834 - Epoch 17 completed in 18.43s
2024-12-10 16:35:21,834 - Train Loss: 0.0741, Train Acc: 97.71%
2024-12-10 16:35:21,834 - Test Loss: 0.0003, Test Acc: 99.32%
2024-12-10 16:35:21,834 - Train/Test Accuracy Gap: -1.61%
2024-12-10 16:35:21,834 - 
Epoch 18/20
2024-12-10 16:35:21,855 - Train Epoch: 18 [0/60000 (0%)]        Loss: 0.048100
2024-12-10 16:35:23,610 - Train Epoch: 18 [6400/60000 (11%)]    Loss: 0.085143
2024-12-10 16:35:25,406 - Train Epoch: 18 [12800/60000 (21%)]   Loss: 0.180130
2024-12-10 16:35:27,201 - Train Epoch: 18 [19200/60000 (32%)]   Loss: 0.015946
2024-12-10 16:35:28,995 - Train Epoch: 18 [25600/60000 (43%)]   Loss: 0.093972
2024-12-10 16:35:30,829 - Train Epoch: 18 [32000/60000 (53%)]   Loss: 0.044350
2024-12-10 16:35:32,584 - Train Epoch: 18 [38400/60000 (64%)]   Loss: 0.045503
2024-12-10 16:35:34,336 - Train Epoch: 18 [44800/60000 (75%)]   Loss: 0.061864
2024-12-10 16:35:36,142 - Train Epoch: 18 [51200/60000 (85%)]   Loss: 0.048607
2024-12-10 16:35:37,932 - Train Epoch: 18 [57600/60000 (96%)]   Loss: 0.016032
2024-12-10 16:35:38,568 - Training Epoch 18: Average loss: 0.0743, Accuracy: 97.69%
2024-12-10 16:35:39,363 - Test set: Average loss: 0.0004, Accuracy: 9914/10000 (99.14%)
2024-12-10 16:35:39,363 - 
Possible overfitting detected!
2024-12-10 16:35:39,363 - - Train loss decreasing while test loss increasing
2024-12-10 16:35:39,363 - Consider:
1. Increasing dropout rate
2. Adding data augmentation
3. Reducing model complexity
4. Early stopping
2024-12-10 16:35:39,363 - Epoch 18 completed in 17.53s
2024-12-10 16:35:39,363 - Train Loss: 0.0743, Train Acc: 97.69%
2024-12-10 16:35:39,363 - Test Loss: 0.0004, Test Acc: 99.14%
2024-12-10 16:35:39,363 - Train/Test Accuracy Gap: -1.45%
2024-12-10 16:35:39,363 - 
Epoch 19/20
2024-12-10 16:35:39,384 - Train Epoch: 19 [0/60000 (0%)]        Loss: 0.233879
2024-12-10 16:35:41,189 - Train Epoch: 19 [6400/60000 (11%)]    Loss: 0.056850
2024-12-10 16:35:42,959 - Train Epoch: 19 [12800/60000 (21%)]   Loss: 0.065966
2024-12-10 16:35:44,735 - Train Epoch: 19 [19200/60000 (32%)]   Loss: 0.039386
2024-12-10 16:35:46,538 - Train Epoch: 19 [25600/60000 (43%)]   Loss: 0.134506
2024-12-10 16:35:48,286 - Train Epoch: 19 [32000/60000 (53%)]   Loss: 0.023465
2024-12-10 16:35:50,059 - Train Epoch: 19 [38400/60000 (64%)]   Loss: 0.043148
2024-12-10 16:35:51,904 - Train Epoch: 19 [44800/60000 (75%)]   Loss: 0.107723
2024-12-10 16:35:53,738 - Train Epoch: 19 [51200/60000 (85%)]   Loss: 0.041184
2024-12-10 16:35:55,605 - Train Epoch: 19 [57600/60000 (96%)]   Loss: 0.017734
2024-12-10 16:35:56,290 - Training Epoch 19: Average loss: 0.0764, Accuracy: 97.59%
2024-12-10 16:35:57,135 - Test set: Average loss: 0.0004, Accuracy: 9911/10000 (99.11%)
2024-12-10 16:35:57,135 - Epoch 19 completed in 17.77s
2024-12-10 16:35:57,135 - Train Loss: 0.0764, Train Acc: 97.59%
2024-12-10 16:35:57,135 - Test Loss: 0.0004, Test Acc: 99.11%
2024-12-10 16:35:57,135 - Train/Test Accuracy Gap: -1.52%
2024-12-10 16:35:57,135 - Early stopping triggered after 19 epochs
2024-12-10 16:35:57,135 - 
Training completed in 5.77 minutes
2024-12-10 16:35:57,135 - Best Test Accuracy: 99.40% achieved at epoch 14
(venv) shriti@Shritis-MacBook-Pro fresh_repo % 
```
