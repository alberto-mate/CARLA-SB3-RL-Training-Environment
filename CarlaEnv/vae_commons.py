import torch
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
    frame = np.transpose(frame, (2, 0, 1))
    frame = frame.astype(np.float32) / 255.0
    frame = torch.tensor(frame)
    frame = torch.unsqueeze(frame, dim=0)
    return frame
def create_encode_state_fn(vae):
    """
        Returns a function that encodes the current state of
        the environment into some feature vector.
    """
    def encode_state(env):
        # Encode image with VAE
        with torch.no_grad():
            frame = preprocess_frame(env.observation)
            mu, logvar = vae.encode(frame)
            encoded_state = vae.reparameterize(mu, logvar).cpu().detach().numpy()
        return encoded_state

    return encode_state