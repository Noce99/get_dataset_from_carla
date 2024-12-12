import numpy as np
import torch

def voxel_grid_stereo_to_rgb(left: torch.Tensor, right: torch.Tensor):
    rgb_left = voxel_grid_mono_to_rgb(left)
    rgb_right = voxel_grid_mono_to_rgb(right)

    big_image = np.ones((left.shape[0],
                         left.shape[1]+right.shape[1], 3))
    big_image[:left.shape[0], :left.shape[1]] = rgb_left
    big_image[  :left.shape[0],
                left.shape[1]:left.shape[1]+right.shape[1]] = rgb_right

    return big_image

def voxel_grid_mono_to_rgb(left: torch.Tensor):
    if left.dim() == 4 and left.shape[0] == 1:
        left = left[0, :, :, :]
    elif left.dim() != 3:
        raise Exception(f"Unexpected shape: {left.shape} (expecting [1, W, H, C] or [W, H, C])!")
    # Now for sure a_voxel_grid have shape [C, W, H]
    # 1) Let's sum along channels
    left = torch.mean(left, 0)
    # Now for sure a_voxel_grid have shape [W, H]

    # 2) Go to Numpy
    left = left.numpy()
    rgb_left = np.ones((left.shape[0], left.shape[1], 3)) * [255, 255, 255]
    rgb_left[left > 0.1] = [0, 0, 255]
    rgb_left[left < - 0.1] = [255, 0, 0]

    return rgb_left

def histogram_stereo_to_rgb(left: torch.Tensor, right: torch.Tensor):
    rgb_left = histogram_mono_to_rgb(left)
    rgb_right = histogram_mono_to_rgb(right)

    big_image = np.ones((left.shape[0],
                         left.shape[1] + right.shape[1], 3))
    big_image[:left.shape[0], :left.shape[1]] = rgb_left
    big_image[:left.shape[0],
    left.shape[1]:left.shape[1] + right.shape[1]] = rgb_right

    return big_image

def histogram_mono_to_rgb(representation: torch.Tensor):
    if representation.dim() == 4 and representation.shape[0] == 1:
        representation = representation[0, :, :, :]
    elif representation.dim() != 3:
        raise Exception(f"Unexpected shape: {representation.shape} (expecting [1, W, H, C] or [W, H, C])!")
    # Now for sure representation have shape [2, W, H]
    positive_polarity = representation[0, :, :]
    negative_polarity = representation[1, :, :]
    # Now for sure representations have shape [W, H]

    # 2) Go to Numpy
    positive_polarity = positive_polarity.numpy()
    negative_polarity = negative_polarity.numpy()
    rgb_representation = np.ones((positive_polarity.shape[0], positive_polarity.shape[1], 3)) * [255, 255, 255]
    rgb_representation[positive_polarity > 0] = [255, 0, 0]
    rgb_representation[negative_polarity > 0] = [0, 0, 255]
    rgb_representation[positive_polarity > negative_polarity] = [255, 0, 0]

    return rgb_representation