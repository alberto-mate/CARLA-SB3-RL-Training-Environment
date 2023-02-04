from torchvision import transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from models import VAE
from utils.misc import LSIZE

source_shape = (80, 160, 3)
target_shape = (80, 160, 3)

z_range = 10
vae_dir = f'/vae/log_dir/vae_{LSIZE}'
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
                        sample = torch.tensor(z).to(device)
                        sample = model.decode(sample).cpu()
                        generated_image = sample.view(3, 80, 160).numpy().transpose((1, 2, 0))

                    compound_image[:, j * w:(j + 1) * w, :] = generated_image
                ax[i, k].imshow(compound_image,vmin=0, vmax=1)
                ax[i, k].set_xticks(np.linspace(w / 2, w * 5 - w / 2, 5))
                ax[i, k].set_xticklabels(np.linspace(-z_range, z_range, 5))
                ax[i, k].set_yticks([h / 2])
                ax[i, k].set_yticklabels([z_index])
        fig.text(0.04, 0.5, "z index", va="center", rotation="vertical")
        fig.suptitle(f'VAE {LSIZE}')
        plt.savefig(os.path.join(vae_dir, 'plot_vae.png'), dpi=700)
        print("ploted")


if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    torch.manual_seed(643)
    # Fix numeric divergence due to bug in Cudnn
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if cuda else "cpu")
    # Load the VAE model
    model = VAE(LSIZE).to(device)

    model_dir = os.path.join(vae_dir, 'best.tar')
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
    image_path = '/CarlaEnv/vae/images/rgb/600.png'  # path to the sample image



    visualizer = VAEVisualizer(model, device, image_path)
    visualizer.visualize()