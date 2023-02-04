""" Various auxiliary utilities """
import math
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
import gym

# A bit dirty: manually change size of car racing env

# Hardcoded for now
LSIZE = 32



def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)
