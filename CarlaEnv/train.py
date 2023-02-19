from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from CarlaEnv.envs.carla_route_vae_env import CarlaRouteEnv
import time
from vae.utils.misc import LSIZE
from vae_commons import create_encode_state_fn, load_vae

from rewards import reward_fn
from utils import HParamCallback, TensorboardCallback

from config import CONFIG

log_dir = 'tensorboard/'

algorithm_dict = {"PPO": PPO, "DQN": DQN}

if CONFIG["algorithm"] not in algorithm_dict:
    raise ValueError("Invalid algorithm name")

AlgorithmRL = algorithm_dict[CONFIG["algorithm"]]

vae = load_vae(f'../vae/log_dir/vae_{LSIZE}', LSIZE)
observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(vae, ["steer", "throttle", "speed",
                                                                                 "angle_next_waypoint", "maneuver"])

env = CarlaRouteEnv(obs_res=(160, 80),
                    reward_fn=reward_fn,
                    observation_space = observation_space,
                    encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
                    start_carla=True, fps=15, action_smoothing=CONFIG["action_smoothing"],
                    action_space_type='continuous', activate_spectator=False)

model = AlgorithmRL('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir, device='cpu',
                    **CONFIG["algorithm_params"])
model_name = f'{model.__class__.__name__}_VAE{LSIZE}_{time.time()}'
new_logger = configure(log_dir + model_name, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
total_timesteps = 500_000
model.learn(total_timesteps=total_timesteps, callback=[HParamCallback(CONFIG), TensorboardCallback(1), CheckpointCallback(
    save_freq=total_timesteps // 10,
    save_path=log_dir + model_name)])
