import torch
from torchvision import transforms

import os
from vae.models import VAE
import numpy as np
import gym
from config import CONFIG

from vae.utils.misc import LSIZE
from carla_env.wrappers import vector, get_displacement_vector

torch.cuda.empty_cache()


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


def create_encode_state_fn(vae, measurements_to_include):
    """
        Returns a function that encodes the current state of
        the environment into some feature vector.
    """

    measure_flags = ["steer" in measurements_to_include,
                     "throttle" in measurements_to_include,
                     "speed" in measurements_to_include,
                     "angle_next_waypoint" in measurements_to_include,
                     "maneuver" in measurements_to_include,
                     "waypoints" in measurements_to_include,
                     "rgb_camera" in measurements_to_include,
                     "seg_camera" in measurements_to_include]

    def create_observation_space():
        observation_space = {}
        if vae: observation_space['vae_latent'] = gym.spaces.Box(low=-4, high=4, shape=(1, LSIZE), dtype=np.float32)
        low, high = [], []
        if measure_flags[0]: low.append(-1), high.append(1)
        if measure_flags[1]: low.append(0), high.append(1)
        if measure_flags[2]: low.append(0), high.append(120)
        if measure_flags[3]: low.append(-3.14), high.append(3.14)
        observation_space['vehicle_measures'] = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

        if measure_flags[4]: observation_space['maneuver'] = gym.spaces.Discrete(4)

        if measure_flags[5]: observation_space['waypoints'] = gym.spaces.Box(low=-50, high=50, shape=(15, 2),
                                                                             dtype=np.float32)

        if measure_flags[6]: observation_space['rgb_camera'] = gym.spaces.Box(low=0, high=255, shape=(CONFIG['obs_res'][1], CONFIG['obs_res'][0], 3), dtype=np.uint8)
        if measure_flags[7]: observation_space['seg_camera'] = gym.spaces.Box(low=0, high=255, shape=(CONFIG['obs_res'][1], CONFIG['obs_res'][0], 3), dtype=np.uint8)
        return gym.spaces.Dict(observation_space)

    def encode_state(env):
        encoded_state = {}
        if vae:
            with torch.no_grad():
                frame = preprocess_frame(env.observation)
                mu, logvar = vae.encode(frame)
                vae_latent = vae.reparameterize(mu, logvar)[0].cpu().detach().numpy()[np.newaxis, :]
            encoded_state['vae_latent'] = vae_latent
        vehicle_measures = []
        if measure_flags[0]: vehicle_measures.append(env.vehicle.control.steer)
        if measure_flags[1]: vehicle_measures.append(env.vehicle.control.throttle)
        if measure_flags[2]: vehicle_measures.append(env.vehicle.get_speed())
        if measure_flags[3]: vehicle_measures.append(env.vehicle.get_angle(env.current_waypoint))
        encoded_state['vehicle_measures'] = vehicle_measures
        if measure_flags[4]: encoded_state['maneuver'] = env.current_road_maneuver.value

        if measure_flags[5]:
            next_waypoints_state = env.route_waypoints[env.current_waypoint_index: env.current_waypoint_index + 15]
            waypoints = [vector(way[0].transform.location) for way in next_waypoints_state]

            vehicle_location = vector(env.vehicle.get_location())
            theta = np.deg2rad(env.vehicle.get_transform().rotation.yaw)

            relative_waypoints = np.zeros((15, 2))
            for i, w_location in enumerate(waypoints):
                relative_waypoints[i] = get_displacement_vector(vehicle_location, w_location, theta)[:2]
            if len(waypoints) < 15:
                start_index = len(waypoints)
                reference_vector = relative_waypoints[start_index-1] - relative_waypoints[start_index-2]
                for i in range(start_index, 15):
                    relative_waypoints[i] = relative_waypoints[i-1] + reference_vector

            encoded_state['waypoints'] = relative_waypoints

        if measure_flags[6]: encoded_state['rgb_camera'] = env.observation
        if measure_flags[7]: encoded_state['seg_camera'] = env.observation

        return encoded_state

    def decode_vae_state(z):
        with torch.no_grad():
            sample = torch.tensor(z)
            sample = vae.decode(sample).cpu()
            generated_image = sample.view(3, 80, 160).numpy().transpose((1, 2, 0)) * 255
        return generated_image
    if not vae:
        decode_state = None
    return create_observation_space(), encode_state, decode_vae_state
