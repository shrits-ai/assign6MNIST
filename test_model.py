import unittest
import torch
import torch.nn as nn
from model import Net
from utils import get_data_loaders
from train import main
import inspect

class TestModelArchitecture(unittest.TestCase):
    def setUp(self):
        self.model = Net()
        self.train_loader, _ = get_data_loaders()

    def test_parameter_count(self):
        """Test if model has less than 20k parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertLess(total_params, 20000, 
                       f'Model has {total_params} parameters, should be less than 20000')
        print(f'Total parameters: {total_params}')

    def test_batch_normalization(self):
        """Test if model uses batch normalization"""
        has_bn = any(isinstance(m, nn.BatchNorm2d) for m in self.model.modules())
        self.assertTrue(has_bn, 'Model should use Batch Normalization')
        
        # Count number of BatchNorm layers
        bn_count = sum(1 for m in self.model.modules() if isinstance(m, nn.BatchNorm2d))
        print(f'Number of BatchNorm layers: {bn_count}')

    def test_dropout(self):
        """Test if model uses dropout"""
        has_dropout = any(isinstance(m, nn.Dropout) for m in self.model.modules())
        self.assertTrue(has_dropout, 'Model should use Dropout')
        
        # Get dropout rate
        dropout_rates = [m.p for m in self.model.modules() if isinstance(m, nn.Dropout)]
        print(f'Dropout rates used: {dropout_rates}')

    def test_fc_or_gap(self):
        """Test if model has either FC layer or Global Average Pooling"""
        has_fc = any(isinstance(m, nn.Linear) for m in self.model.modules())
        has_gap = 'adaptive_avg_pool2d' in str(self.model.forward).lower() or \
                 'AdaptiveAvgPool2d' in str(inspect.getsource(Net))
        
        self.assertTrue(has_fc or has_gap, 
                       'Model should have either Fully Connected layer or Global Average Pooling')
        print(f'Uses FC layer: {has_fc}, Uses GAP: {has_gap}')

    def test_epoch_count(self):
        """Test if number of epochs is less than or equal to 20"""
        with open('train.py', 'r') as f:
            train_code = f.read()
        
        # Check for epochs definition in main()
        self.assertIn('epochs', train_code)
        # Extract epochs value (this is a simple check, might need adjustment based on code structure)
        epochs = None
        for line in train_code.split('\n'):
            if 'epochs' in line and '=' in line:
                try:
                    epochs = int(line.split('=')[1].strip())
                    break
                except:
                    continue
        
        self.assertIsNotNone(epochs, 'Could not find epochs definition')
        self.assertLessEqual(epochs, 20, f'Number of epochs ({epochs}) should be less than or equal to 20')
        print(f'Number of epochs: {epochs}')

    def test_data_augmentation(self):
        """Test if data augmentation is used in training"""
        # Check if transforms are used in data loading
        first_batch = next(iter(self.train_loader))
        self.assertIsInstance(first_batch[0], torch.Tensor, 
                            'Data loader should return tensors')
        
        with open('utils.py', 'r') as f:
            utils_code = f.read()
        
        # Check for common augmentation transforms
        augmentation_keywords = [
            'RandomRotation', 'RandomAffine', 'RandomCrop', 'RandomResizedCrop',
            'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomPerspective',
            'ColorJitter'
        ]
        
        has_augmentation = any(keyword in utils_code for keyword in augmentation_keywords)
        self.assertTrue(has_augmentation, 
                       'No data augmentation found in transforms')
        print('Data augmentation check passed')

if __name__ == '__main__':
    unittest.main(verbosity=2) 