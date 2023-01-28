import numpy as np 
import matplotlib.pyplot as plt 
from skimage import io
from torch.nn import functional as F
import torch


img = np.array(range(10000)).reshape(100, 100)

tensor_img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
size = tensor_img.shape

angle = 0 * np.pi / 180
translation = np.array([-0.5, 0.5])

theta = np.array([np.cos(angle), np.sin(-angle), translation[0], 
				  np.sin(angle), np.cos(angle), translation[1]]).reshape(1, 2, 3)


theta = torch.tensor(theta)
grid = F.affine_grid(theta, torch.Size(size), align_corners=True).float()

output_img = F.grid_sample(tensor_img, grid, mode='nearest', align_corners=True)


output_img = output_img.squeeze(0).squeeze(0).numpy()

plt.imshow(output_img)
plt.show()