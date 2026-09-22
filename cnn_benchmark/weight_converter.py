"""
PyTorch state_dict → custom nn library weight converter.

Maps PyTorch tensor layouts to nn library layouts:
- Conv2d: (O, I, H, W) → (H, W, I, O)
- Linear: (O, I) → (I, O)
- BatchNorm: handled if present
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
from nn.autograd import Tensor


def pytorch_to_nn_conv2d(weight: np.ndarray, bias: Optional[np.ndarray]) -> Tuple[Tensor, Tensor]:
    """
    Convert PyTorch Conv2d weights to nn.Conv2D format.
    
    PyTorch: weight (out_channels, in_channels, kh, kw), bias (out_channels,)
    nn lib:  kernel (kh, kw, in_channels, out_channels), bias (out_channels,)
    """
    # Transpose: (O, I, H, W) -> (H, W, I, O)
    kernel = weight.transpose(2, 3, 1, 0).astype(np.float32)
    kernel_t = Tensor(kernel, requires_grad=True)
    
    if bias is not None:
        bias_t = Tensor(bias.astype(np.float32), requires_grad=True)
    else:
        bias_t = Tensor(np.zeros(weight.shape[0], dtype=np.float32), requires_grad=True)
    
    return kernel_t, bias_t


def pytorch_to_nn_linear(weight: np.ndarray, bias: Optional[np.ndarray]) -> Tuple[Tensor, Tensor]:
    """
    Convert PyTorch Linear weights to nn.Dense format.
    
    PyTorch: weight (out_features, in_features), bias (out_features,)
    nn lib:  W (in_features, out_features), b (out_features,)
    """
    W = weight.T.astype(np.float32)  # (I, O)
    W_t = Tensor(W, requires_grad=True)
    
    if bias is not None:
        b_t = Tensor(bias.astype(np.float32), requires_grad=True)
    else:
        b_t = Tensor(np.zeros(weight.shape[0], dtype=np.float32), requires_grad=True)
    
    return W_t, b_t


def load_pytorch_state_dict(state_dict: Dict[str, np.ndarray], 
                            model,
                            layer_mapping: Dict[str, str]) -> None:
    """
    Load PyTorch state_dict into nn model using layer name mapping.
    
    Args:
        state_dict: PyTorch state_dict (converted to numpy)
        model: nn model with layers accessible by name
        layer_mapping: {pytorch_layer_name: nn_layer_attribute_path}
                       e.g., {"conv1": "seq.layers.0", "fc1": "seq.layers.3"}
    """
    for pt_name, nn_attr in layer_mapping.items():
        weight_key = f"{pt_name}.weight"
        bias_key = f"{pt_name}.bias"
        
        if weight_key not in state_dict:
            continue
            
        weight = state_dict[weight_key]
        bias = state_dict.get(bias_key)
        
        # Navigate to nn layer
        layer = model
        for part in nn_attr.split('.'):
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        
        # Convert and assign based on layer type
        if hasattr(layer, 'kernel') and hasattr(layer, 'bias'):  # Conv2D
            kernel_t, bias_t = pytorch_to_nn_conv2d(weight, bias)
            layer.kernel = kernel_t
            layer.bias = bias_t
            layer.built = True
        elif hasattr(layer, 'W') and hasattr(layer, 'b'):  # Dense
            W_t, b_t = pytorch_to_nn_linear(weight, bias)
            layer.W = W_t
            layer.b = b_t
            layer.built = True
        else:
            print(f"Warning: Unknown layer type for {pt_name} -> {nn_attr}")


def build_custom_mnist_model():
    """Build the custom MNIST model from mnist_test.py for weight loading."""
    from nn.activations import relu, softmax
    from nn.layers import Conv2D, Dense, Dropout, Flatten
    from nn.models import Sequential
    
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding="valid"))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding="valid"))
    model.add(Flatten())
    model.add(Dense(128, activation=relu))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation=softmax))
    
    return model


def build_pytorch_mnist_model():
    """Build equivalent PyTorch model for MNIST (matching custom architecture)."""
    import torch
    import torch.nn as nn
    
    class MNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 24 * 24, 128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 10)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.flatten(x)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    return MNISTNet()


def get_custom_to_pytorch_mapping() -> Dict[str, str]:
    """Mapping from custom nn layer names to PyTorch layer names."""
    return {
        "conv1": "seq.layers.0",
        "conv2": "seq.layers.1",
        "fc1": "seq.layers.3",
        "fc2": "seq.layers.5",
    }


def get_pytorch_to_custom_mapping() -> Dict[str, str]:
    """Mapping from PyTorch layer names to custom nn layer paths."""
    return {
        "conv1": "seq.layers.0",
        "conv2": "seq.layers.1", 
        "fc1": "seq.layers.3",
        "fc2": "seq.layers.5",
    }


def convert_and_save_mnist_weights(pytorch_weights_path: str, output_path: str):
    """Convert saved PyTorch MNIST model weights to nn format."""
    import torch
    
    # Load PyTorch model
    pt_model = build_pytorch_mnist_model()
    pt_model.load_state_dict(torch.load(pytorch_weights_path, map_location='cpu'))
    pt_state_dict = {k: v.numpy() for k, v in pt_model.state_dict().items()}
    
    # Build custom model
    custom_model = build_custom_mnist_model()
    # Initialize with dummy forward to build layers
    import numpy as np
    dummy = np.random.randn(1, 28, 28, 1).astype(np.float32)
    custom_model(dummy)
    
    # Load weights
    mapping = get_pytorch_to_custom_mapping()
    load_pytorch_state_dict(pt_state_dict, custom_model, mapping)
    
    # Save custom model weights
    import pickle
    weights_dict = {}
    for name, param in custom_model.parameters():
        weights_dict[name] = param.data
    
    with open(output_path, 'wb') as f:
        pickle.dump(weights_dict, f)
    
    print(f"Converted weights saved to {output_path}")
    return custom_model


def load_custom_weights(model, weights_path: str):
    """Load converted weights into custom nn model."""
    import pickle
    with open(weights_path, 'rb') as f:
        weights_dict = pickle.load(f)
    
    # Map weights back to model parameters
    for name, param in model.parameters():
        if name in weights_dict:
            param.data = weights_dict[name]
    
    return model


def download_pretrained_pytorch(model_name: str) -> Dict[str, np.ndarray]:
    """
    Download and return PyTorch pretrained weights as numpy arrays.
    Supports: 'resnet18', 'alexnet', 'vgg16', 'googlenet', 'mobilenet_v2', 'densenet121', 'squeezenet1_0'
    """
    import torch
    import torchvision.models as models
    
    model_map = {
        'resnet18': models.resnet18,
        'alexnet': models.alexnet,
        'vgg16': models.vgg16,
        'googlenet': models.googlenet,
        'mobilenet_v2': models.mobilenet_v2,
        'densenet121': models.densenet121,
        'squeezenet1_0': models.squeezenet1_0,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    model_fn = model_map[model_name]
    model = model_fn(weights='DEFAULT')
    state_dict = {k: v.numpy() for k, v in model.state_dict().items()}
    return state_dict


if __name__ == "__main__":
    # Quick test
    import torch
    pt_model = build_pytorch_mnist_model()
    print("PyTorch model state_dict keys:")
    for k, v in pt_model.state_dict().items():
        print(f"  {k}: {v.shape}")
    
    custom_model = build_custom_mnist_model()
    dummy = np.random.randn(1, 28, 28, 1).astype(np.float32)
    custom_model(dummy)
    print("\nCustom model parameters:")
    for name, param in custom_model.parameters():
        print(f"  {name}: {param.data.shape}")