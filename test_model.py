import torch
from utils import Net
from torchsummary import summary
import torch.nn.functional as F

def test_parameter_count():
    print("\nRunning Parameter Count Test...")
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test if parameter count is within limit
    assert total_params < 20000, f"Model has too many parameters: {total_params}"
    assert total_params == trainable_params, "All parameters should be trainable"
    print("✅ Parameter count test passed!")
    return True

def test_batch_norm_usage():
    print("\nRunning BatchNorm Usage Test...")
    model = Net()
    
    # Check if BatchNorm layers exist
    has_bn = any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    assert has_bn, "Model should use BatchNorm layers"
    
    # Test BatchNorm configuration
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for idx, bn in enumerate(bn_layers):
        assert bn.momentum == 0.1, f"BatchNorm{idx+1} has incorrect momentum"
        assert bn.eps == 1e-5, f"BatchNorm{idx+1} has incorrect epsilon"
        assert bn.training, f"BatchNorm{idx+1} should be in training mode by default"
    
    print(f"✅ Found {len(bn_layers)} BatchNorm layers with correct configuration!")
    return True

def test_dropout_usage():
    print("\nRunning Dropout Usage Test...")
    model = Net()
    
    # Check if Dropout layers exist
    dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropout_layers) > 0, "Model should use Dropout"
    
    # Test Dropout rate
    for idx, dropout in enumerate(dropout_layers):
        assert 0 < dropout.p <= 0.5, f"Dropout{idx+1} rate should be between 0 and 0.5"
    
    # Test if Dropout is active in training mode
    model.train()
    test_input = torch.randn(10, 1, 28, 28)
    out1 = model(test_input)
    out2 = model(test_input)
    assert not torch.equal(out1, out2), "Dropout should give different outputs in training mode"
    
    # Test if Dropout is inactive in eval mode
    model.eval()
    out1 = model(test_input)
    out2 = model(test_input)
    assert torch.equal(out1, out2), "Dropout should give same outputs in eval mode"
    
    print(f"✅ Found {len(dropout_layers)} Dropout layers with correct configuration!")
    return True

def test_final_layer():
    print("\nRunning Final Layer Test...")
    model = Net()
    
    # Check if model uses FC or GAP
    has_fc = any(isinstance(module, torch.nn.Linear) for module in model.modules())
    has_gap = any(isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in model.modules())
    
    assert has_fc != has_gap, "Model should use either FC or GAP, not both"
    
    if has_fc:
        fc_layer = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
        assert fc_layer.out_features == 10, "Final FC layer should output 10 classes"
        print("✅ Model uses Fully Connected layer as final layer")
    else:
        gap_layer = [m for m in model.modules() if isinstance(m, torch.nn.AdaptiveAvgPool2d)][-1]
        print("✅ Model uses Global Average Pooling as final layer")
    
    return True

def test_model_architecture():
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    print("\nRunning Complete Model Architecture Tests:")
    print("=========================================")
    
    # Run all tests
    try:
        test_parameter_count()
        test_batch_norm_usage()
        test_dropout_usage()
        test_final_layer()
        
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
        
        # Test model training mode
        model.train()
        train_output = model(test_batch)
        model.eval()
        eval_output = model(test_batch)
        assert not torch.equal(train_output, eval_output), "Dropout should make training and eval outputs different"
        
        print("\n✅ All architecture tests passed successfully!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

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