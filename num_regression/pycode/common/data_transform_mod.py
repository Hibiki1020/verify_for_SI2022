from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms
from torchvision.transforms.transforms import Resize
import torch.nn.functional as nn_functional

class DataTransform():
    def __init__(self, resize, mean, std):
        self.mean = mean
        self.std = std
        size = (resize, resize)

        self.img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

    def __call__(self, img_pil, deg_numpy, phase="train"):
        ## img: numpy -> tensor
        img_tensor = self.img_transform(img_pil)
        
        deg_numpy = deg_numpy.astype(np.float32)
        deg_numpy = deg_numpy / np.linalg.norm(deg_numpy)
        deg_tensor = torch.from_numpy(deg_numpy)
        return img_tensor, deg_tensor