import torch
from torchvision import transforms

import os
from vae.models import VAE
import numpy as np

torch.cuda.empty_cache()


# vae_dir = f'/home/albertomate/Documentos/carla/PythonAPI/my-carla/vae/log_dir/vae_{LSIZE}'
def load_vae(vae_dir, latent_size):
    model_dir = os.path.join(vae_dir, 'best.tar')
    model = VAE(latent_size)
    if os.path.exists(model_dir):
        state = torch.load(model_dir)
        print("Reloading model at epoch {}"
              ", with test error {}".format(
            state['epoch'],
            state['precision']))
        model.load_state_dict(state['state_dict'])
        return model
    raise "Error - VAE model does not exist"

def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    frame = preprocess(frame).unsqueeze(0)
    return frame
def create_encode_state_fn(vae):
    """
        Returns a function that encodes the current state of
        the environment into some feature vector.
    """
    def encode_state(env):
        # Encode image with VAE
        # preprocess = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        # image = preprocess(env.observation).unsqueeze(0)
        with torch.no_grad():
            frame = preprocess_frame(env.observation)
            mu, logvar = vae.encode(frame)
            encoded_state = vae.reparameterize(mu, logvar)[0].cpu().detach().numpy()
        return encoded_state

    def decode_state(z):
        with torch.no_grad():
            sample = torch.tensor(z)
            sample = vae.decode(sample).cpu()
            generated_image = sample.view(3, 80, 160).numpy().transpose((1, 2, 0)) * 255
        return generated_image


    return encode_state, decode_state