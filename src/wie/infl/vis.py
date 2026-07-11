import matplotlib.pyplot as plt


def show_image(tensor):
    """
    Input: tensor of shape [1, 1, 28, 28] or [1, 28, 28]
    Function: Display as an image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)  # Remove channel dimension

    plt.imshow(tensor.cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()


def compute_norm_tensor_list(tensor_list):
    """
    Input: A list containing tensors
    Function: Calculate the L2 norm of each tensor
    Output: A list containing the L2 norm of each tensor
    """
    return [tensor.norm() for tensor in tensor_list]


def sum_norm(tensor_list):
    """
    Input: A list containing tensors
    Function: Calculate the L2 norm of each tensor
    Output: A list containing the L2 norm of each tensor
    """
    return sum(compute_norm_tensor_list(tensor_list))
