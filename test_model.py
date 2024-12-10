import torch
from utils import Net
from torchsummary import summary

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
    
    print("\nModel architecture tests passed!")
    return True

if __name__ == "__main__":
    test_model_architecture() 