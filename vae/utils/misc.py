""" Various auxiliary utilities """
import math
from os.path import join, exists
import random

import torch
from torchvision import transforms
import numpy as np
import gym
import cv2
from PIL import Image, ImageFilter

# A bit dirty: manually change size of car racing env

# Hardcoded for now
LSIZE = 64



def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


class RandomMotionBlur:
    def __init__(self, p=0.5, kernel_size=20, angle_range=(-10, 10)):
        self.angle_range = angle_range
        self.probability = p

        self.kernel_size = kernel_size

    def __call__(self, image):
        if np.random.rand() < self.probability:
            angle = random.randint(*self.angle_range)
            # Define the motion blur kernel
            kernel = np.zeros((self.kernel_size, self.kernel_size))
            kernel[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size)
            kernel /= self.kernel_size

            # Rotate the kernel
            kernel = self._rotate_kernel(kernel, angle)

            # Apply the motion blur filter
            return image.filter(ImageFilter.Kernel((self.kernel_size, self.kernel_size), kernel.flatten()))
        else:
            return image

    def _rotate_kernel(self, kernel, angle):
        kernel_center = tuple(np.array(kernel.shape) / 2)
        kernel_size = kernel.shape[0]
        rot_mat = cv2.getRotationMatrix2D(kernel_center, angle, 1.0)
        kernel_rotated = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))
        return kernel_rotated
