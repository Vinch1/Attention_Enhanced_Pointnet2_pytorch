"""
Device utility for cross-platform PyTorch support.
Automatically detects and uses the best available device:
- MPS (Metal Performance Shaders) for Apple Silicon
- CUDA for NVIDIA GPUs
- CPU fallback
"""
import torch


def get_device(use_cpu=False):
    """
    Get the best available device for PyTorch computations.

    Args:
        use_cpu: If True, force CPU usage regardless of GPU availability

    Returns:
        torch.device: The device to use
    """
    if use_cpu:
        return torch.device('cpu')

    # Check for Apple Silicon MPS (Metal Performance Shaders)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')

    # Check for CUDA
    if torch.cuda.is_available():
        return torch.device('cuda')

    # Fallback to CPU
    return torch.device('cpu')


def get_device_name(device):
    """Get a human-readable name for the device."""
    if device.type == 'mps':
        return "Apple Silicon (MPS)"
    elif device.type == 'cuda':
        return f"NVIDIA GPU (CUDA: {torch.cuda.get_device_name(0)})"
    else:
        return "CPU"


def to_device(data, device):
    """
    Move tensor or dict/tuple of tensors to the specified device.

    Args:
        data: Tensor, dict, tuple, or list of tensors
        device: Target device

    Returns:
        Data moved to the specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (tuple, list)):
        return type(data)(to_device(v, device) for v in data)
    return data
