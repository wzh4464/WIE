import torch
import numpy as np
import argparse
import warnings

# no future warning
warnings.simplefilter(action="ignore", category=FutureWarning)


def get_tensor_size(tensor):
    """Calculate the size of a torch.Tensor (in bytes)"""
    return tensor.numel() * tensor.element_size()


def get_size(obj, seen=None):
    """Recursively calculate the size of an object, considering torch.Tensor and numpy.ndarray"""
    size = 0
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, torch.Tensor):
        size += get_tensor_size(obj)
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes
    elif isinstance(obj, dict):
        for key, value in obj.items():
            size += get_size(key, seen)
            size += get_size(value, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += get_size(item, seen)
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            size += get_size(item, seen)
    return size


# def analyze_netlist(netlist):
#     """Analyze the internal structure of a NetList object"""
#     print("Analyzing NetList object:")
#     for attr in dir(netlist):
#         if attr.startswith('_'):
#             continue  # Skip private attributes
#         try:
#             value = getattr(netlist, attr)
#             size = get_size(value)
#             print(f"  Attribute: {attr}, Size: {size / (1024 ** 2):.2f} MB")
#         except Exception as e:
#             print(f"  Attribute: {attr}, Error accessing: {e}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze a PyTorch checkpoint file")
    parser.add_argument("path", type=str, help="Path to the checkpoint file")
    args = parser.parse_args()

    # Load file
    checkpoint = torch.load(args.path, map_location="cpu")  # Ensure loading on CPU

    # Initialize total size
    total_file_size = 0

    # Calculate the size of each key and accumulate the total size
    print("\nDetailed analysis of all keys:")
    for key, value in checkpoint.items():
        size = get_size(value)
        total_file_size += size
        print(f"Key: {key}, Type: {type(value)}, Size: {size / (1024**3):.2f} GB")

    # Output total size
    print(
        f"\nTotal size calculated from checkpoint: {total_file_size / (1024**3):.2f} GB"
    )

    # # If 'models' is a NetList object, analyze its internal structure
    # if hasattr(checkpoint['models'], 'models'):
    #     models_obj = checkpoint['models']
    #     analyze_netlist(models_obj)


if __name__ == "__main__":
    main()
