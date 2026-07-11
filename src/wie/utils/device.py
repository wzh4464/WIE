import os
import torch


def get_device(gpu: int) -> str:
    """
    Determine the device to use for training based on available hardware.

    Args:
        gpu: GPU index to use

    Returns:
        str: Device identifier ('mps', 'cuda:X', or 'cpu')
    """
    # Check for MPS (Metal Performance Shaders) on macOS
    if torch.backends.mps.is_available():
        return "mps"
    # Check for CUDA. A negative gpu index means "no GPU": guarding on
    # ``gpu >= 0`` avoids building an invalid device string like "cuda:-1".
    elif gpu >= 0 and torch.cuda.is_available():
        return f"cuda:{gpu}"
    # Fall back to CPU (includes the gpu < 0 / "no GPU requested" case).
    else:
        return "cpu"


def get_real_gpu_index(args_gpu: int) -> int:
    """
    Get the real GPU index based on environment variables and command line arguments.

    Args:
        args_gpu: GPU index from command line arguments

    Returns:
        int: Real GPU index to use
    """
    # Get environment variable or use command line argument
    real_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", str(args_gpu))

    if isinstance(real_gpu, str) and "," in real_gpu:
        # If environment variable contains multiple GPUs (e.g., "0,1,2"),
        # select the one specified by args_gpu
        gpu_list = real_gpu.split(",")
        if args_gpu < len(gpu_list):
            real_gpu = int(gpu_list[args_gpu])
        else:
            real_gpu = int(gpu_list[0])  # Default to first GPU
    else:
        # If it's a single value, convert to integer
        try:
            real_gpu = int(real_gpu)
        except ValueError:
            real_gpu = 0  # Default value

    return real_gpu
