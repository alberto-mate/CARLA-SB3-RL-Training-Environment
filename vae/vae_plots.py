from torchvision import transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from models import VAE
from vae.utils.misc import LSIZE

source_shape = (80, 160, 3)
target_shape = (80, 160, 3)

z_range = 10
name = 'vae_64_augmentation'
vae_dir = f'/home/albertomate/Documentos/carla/PythonAPI/my-carla/vae/log_dir/{name}'
class VAEVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def visualize(self, image_path):
        # Load the image
        image = plt.imread(image_path)
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
                        sample = torch.tensor(z).to(self.device)
                        sample = self.model.decode(sample).cpu()
                        generated_image = sample.view(3, 80, 160).numpy().transpose((1, 2, 0))

                    compound_image[:, j * w:(j + 1) * w, :] = generated_image
                ax[i, k].imshow(compound_image,vmin=0, vmax=1)
                ax[i, k].set_xticks(np.linspace(w / 2, w * 5 - w / 2, 5))
                ax[i, k].set_xticklabels(np.linspace(-z_range, z_range, 5))
                ax[i, k].set_yticks([h / 2])
                ax[i, k].set_yticklabels([z_index])
        fig.text(0.04, 0.5, "z index", va="center", rotation="vertical")
        fig.suptitle(f'{name} - latent space exploration')
        plt.savefig(os.path.join(vae_dir, 'plot_vae.png'), dpi=700)
        print("ploted")

    def compare_gt_and_pred(self, image_path_list):
        fig, ax = plt.subplots(len(image_path_list), 3)
        for i, image_path in enumerate(image_path_list):
            # Load the image
            rgb_image = plt.imread(image_path)
            seg_image = plt.imread(image_path.replace('rgb', 'segmentation'))

            # Define the transformations to apply to the images
            # Preprocess the image
            preprocess = transforms.Compose([
                transforms.ToTensor(),
            ])
            rgb_image = preprocess(rgb_image).unsqueeze(0).to(self.device)
            seg_image = preprocess(seg_image).unsqueeze(0).to(self.device)

            # Obtain the latent representation
            with torch.no_grad():
                mu, logvar = self.model.encode(rgb_image)
                seeded_z = self.model.reparameterize(mu, logvar)[0].cpu().detach().numpy()

            with torch.no_grad():
                sample = torch.tensor(seeded_z).to(self.device)
                sample = self.model.decode(sample).cpu()
                generated_image = sample.view(3, 80, 160).numpy().transpose((1, 2, 0))

            ax[i, 0].imshow(rgb_image[0].cpu().numpy().transpose((1, 2, 0)), vmin=0, vmax=1)
            ax[i, 1].imshow(generated_image, vmin=0, vmax=1)
            ax[i, 2].imshow(seg_image[0].cpu().numpy().transpose((1, 2, 0)), vmin=0, vmax=1)

            ax[i, 0].set_axis_off()
            ax[i, 1].set_axis_off()
            ax[i, 2].set_axis_off()

        pad = 5  # in points
        for ax, col in zip(ax[0], ['rgb', 'pred', 'gt']):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='small', ha='center', va='baseline')
        fig.suptitle(f'{name} - compare gt and pred')
        plt.subplots_adjust(wspace=0.01, hspace=0.1)
        plt.savefig(os.path.join(vae_dir, 'compare_gt_and_pred_plot.png'), dpi=700)
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
    image_path = '/home/albertomate/Documentos/carla/PythonAPI/my-carla/vae/images/rgb/520.png'  # path to the sample image

    images_numbers = [520, 140, 254, 984]
    image_path_list = [os.path.join('/home/albertomate/Documentos/carla/PythonAPI/my-carla/vae/images/rgb/', f'{i}.png') for i in images_numbers]
    visualizer = VAEVisualizer(model, device)
    visualizer.visualize(image_path)
    visualizer.compare_gt_and_pred(image_path_list)