import torch
import torch.nn as nn
import torch.optim as optim
from utils import Net, get_data_loaders, setup_logger
import time
import sys

# CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = setup_logger()

def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        running_loss += loss.item()

        if batch_idx % 100 == 0:
            msg = f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ' \
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'
            logger.info(msg)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    logger.info(f'Training Epoch {epoch}: Average loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    return epoch_loss, epoch_acc

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    logger.info(f'Test set: Average loss: {test_loss:.4f}, '
                f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return test_loss, accuracy

def main():
    start_time = time.time()
    logger.info(f"Using device: {device}")
    logger.info("Python version: {}".format(sys.version))
    logger.info("PyTorch version: {}".format(torch.__version__))
    
    # Get data loaders
    logger.info("Loading datasets...")
    train_loader, test_loader = get_data_loaders()
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model, optimizer and criterion
    logger.info("Initializing model...")
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Log model architecture
    logger.info(f"\nModel Architecture:\n{str(model)}")
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Training loop
    logger.info("\nStarting training...")
    epochs = 10
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        epoch_time = time.time() - epoch_start
        
        # Log epoch summary
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f"New best accuracy: {best_acc:.2f}%")
    
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time/60:.2f} minutes")
    logger.info(f"Best Test Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main() 