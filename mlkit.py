import torch


# Function 1: Set device
# Use the fastest computing resource available to run your model
# The function executes a strict hierarchy of checks:
    # Is there an NVIDIA GPU? -> YES: Use cuda (Fastest)
    # Is there an Apple Silicon GPU? -> YES: Use mps (Very Fast on Mac)
    # Otherwise? -> Use cpu (Guaranteed to work)
def choose_device() -> torch.device:
    """Choose the fastest available device that PyTorch can use."""
    if torch.cuda.is_available():  # CUDA means an NVIDIA GPU is available.
        return torch.device("cuda")  # Put tensors and model weights on the NVIDIA GPU.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # Apple GPU check.
        return torch.device("mps")  # Use Apple's Metal backend on supported Macs.
    return torch.device("cpu")  # Fall back to the CPU, which works everywhere.