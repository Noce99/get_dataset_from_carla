import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import torch

def disp_to_rgb(disp_array: np.ndarray):
    if isinstance(disp_array, torch.Tensor):
        disp_array = disp_array.numpy()
    disp_pixels = np.argwhere(disp_array > 0)
    u_indices = disp_pixels[:, 1]
    v_indices = disp_pixels[:, 0]
    disp = disp_array[v_indices, u_indices]
    max_disp = 80

    norm = mpl.colors.Normalize(vmin=0, vmax=max_disp, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='inferno')

    disp_color = mapper.to_rgba(disp)[..., :3]
    output_image = np.zeros((disp_array.shape[0], disp_array.shape[1], 3))
    output_image[v_indices, u_indices, :] = disp_color
    output_image = (255 * output_image).astype("uint8")
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    return output_image