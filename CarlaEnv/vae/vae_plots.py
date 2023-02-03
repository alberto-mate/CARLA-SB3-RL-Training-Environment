import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import os

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import save_image
import cv2
import gzip
import pickle
import torch
from os.path import join, exists
from os import mkdir
from models import VAE

from models import VAE

LSIZE = 32

source_shape = (80, 160, 3)
target_shape = (80, 160, 3)

z_range = 10

class VAEVisualizer:
    def __init__(self, model, device, image_path):
        self.model = model
        self.device = device
        self.image_path = image_path

    def visualize(self):
        # Load the image
        image = plt.imread(self.image_path)
        # Define the transformations to apply to the images
        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = preprocess(image).unsqueeze(0).to(self.device)

        # Obtain the latent representation
        with torch.no_grad():
            mu, logvar = self.model.encode(image)
            seeded_z = self.model.reparameterize(mu, logvar)[0].cpu().detach().numpy()

        fig, ax = plt.subplots(min(16, LSIZE), int(np.ceil(LSIZE / 16)), sharex=True, figsize=(12, 12))

        if len(ax.shape) == 1:
            ax = np.expand_dims(ax, axis=-1)
        for k in range(int(np.ceil(LSIZE / 16))):
            for i in range(16):
                z_index = i + k * 16
                if z_index >= LSIZE:
                    break
                w = source_shape[1]
                h = source_shape[0]
                compound_image = np.zeros((h, w * 5, 3))
                for j, zi in enumerate(np.linspace(-z_range, z_range, 5)):
                    z = seeded_z.copy()
                    z[z_index] += zi
                    with torch.no_grad():
                        sample = torch.randn(1, LSIZE).to(device)
                        sample = model.decode(sample).cpu()
                        sample = sample.view(3, 80, 160).cpu().numpy()
                        generated_image = sample.transpose((1, 2, 0))


                    compound_image[:, j * w:(j + 1) * w, :] = generated_image
                ax[i, k].imshow(compound_image,vmin=0, vmax=1)
                ax[i, k].set_xticks(np.linspace(w / 2, w * 5 - w / 2, 5))
                ax[i, k].set_xticklabels(np.linspace(-z_range, z_range, 5))
                ax[i, k].set_yticks([h / 2])
                ax[i, k].set_yticklabels([z_index])
        fig.text(0.04, 0.5, "z index", va="center", rotation="vertical")
        fig.suptitle('PLOT')
        plt.savefig('plot.png', dpi=700)
        print("ploted")


if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    torch.manual_seed(643)
    # Fix numeric divergence due to bug in Cudnn
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if cuda else "cpu")
    # Load the VAE model
    model = VAE(32).to(device)
    model_dir = '/home/albertomate/Documentos/carla/PythonAPI/my-carla/CarlaEnv/vae/log_dir/vae_1675450899.346522/best.tar'

    if os.path.exists(model_dir):
        state = torch.load(model_dir)
        print("Reloading model at epoch {}"
              ", with test error {}".format(
            state['epoch'],
            state['precision']))
        model.load_state_dict(state['state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Initialize the VAE visualizer
    image_path = '/home/albertomate/Documentos/carla/PythonAPI/my-carla/CarlaEnv/images/rgb/0.png'  # path to the sample image



    visualizer = VAEVisualizer(model, device, image_path)
    visualizer.visualize()