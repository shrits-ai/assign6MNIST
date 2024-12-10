import torch
from utils import Net
from torchsummary import summary
import torch.nn.functional as F

def test_model_architecture():
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    # Print model summary
    print("\nModel Architecture Summary:")
    summary(model, input_size=(1, 28, 28))
    
    # Basic architecture checks
    assert hasattr(model, 'conv1'), "Model missing conv1 layer"
    assert hasattr(model, 'bn1'), "Model missing bn1 layer"
    assert hasattr(model, 'dropout'), "Model missing dropout layer"
    assert isinstance(model.dropout.p, float), "Dropout rate should be float"
    assert 0 <= model.dropout.p <= 1, "Dropout rate should be between 0 and 1"
    
    # Test layer dimensions
    test_input = torch.randn(1, 1, 28, 28).to(device)
    
    # Test Conv1 + BN1 + Pool1
    x = model.conv1(test_input)
    assert x.shape == (1, 8, 28, 28), f"Conv1 output shape incorrect: {x.shape}"
    x = model.bn1(x)
    assert x.shape == (1, 8, 28, 28), f"BN1 output shape incorrect: {x.shape}"
    x = model.pool1(x)
    assert x.shape == (1, 8, 14, 14), f"Pool1 output shape incorrect: {x.shape}"
    
    # Test Conv2 + BN2
    x = model.conv2(x)
    assert x.shape == (1, 16, 14, 14), f"Conv2 output shape incorrect: {x.shape}"
    x = model.bn2(x)
    assert x.shape == (1, 16, 14, 14), f"BN2 output shape incorrect: {x.shape}"
    
    # Test Conv3 + Pool2
    x = model.conv3(x)
    assert x.shape == (1, 16, 14, 14), f"Conv3 output shape incorrect: {x.shape}"
    x = model.pool2(x)
    assert x.shape == (1, 16, 7, 7), f"Pool2 output shape incorrect: {x.shape}"
    
    # Test FC layer
    x = x.view(-1, 16 * 7 * 7)
    assert x.shape == (1, 784), f"Flatten output shape incorrect: {x.shape}"
    x = model.fc1(x)
    assert x.shape == (1, 10), f"FC1 output shape incorrect: {x.shape}"
    
    # Test full forward pass
    test_batch = torch.randn(32, 1, 28, 28).to(device)
    output = model(test_batch)
    assert output.shape == (32, 10), f"Model output shape incorrect: {output.shape}"
    
    # Test output properties
    assert torch.allclose(torch.sum(torch.exp(output), dim=1), 
                         torch.ones(32).to(device), 
                         atol=1e-6), "Output probabilities don't sum to 1"
    
    # Parameter count check
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has too many parameters: {total_params}"
    
    # Test BatchNorm parameters
    assert model.bn1.momentum == 0.1, "Incorrect BatchNorm momentum"
    assert model.bn1.eps == 1e-5, "Incorrect BatchNorm epsilon"
    
    # Test model training mode
    model.train()
    train_output = model(test_batch)
    model.eval()
    eval_output = model(test_batch)
    assert not torch.equal(train_output, eval_output), "Dropout should make training and eval outputs different"
    
    print("\nAll model architecture tests passed!")
    return True

def test_model_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    # Test single training step
    test_input = torch.randn(32, 1, 28, 28).to(device)
    test_target = torch.randint(0, 10, (32,)).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    output = model(test_input)
    loss = F.nll_loss(output, test_target)
    loss.backward()
    optimizer.step()
    
    print("\nModel training test passed!")
    return True

if __name__ == "__main__":
    test_model_architecture()
    test_model_training() 