import os

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from carla_env.envs.carla_route_env import CarlaRouteEnv
import time

from vae.utils.misc import LSIZE
from carla_env.state_commons import create_encode_state_fn, load_vae

from carla_env.rewards import reward_functions
from utils import HParamCallback, TensorboardCallback, write_json

from config import CONFIG

log_dir = 'tensorboard'
#reload_model = "./tensorboard/PPO_vae64_1677524104/model_1400000_steps.zip"
reload_model = ""
total_timesteps = 800_000

seed = CONFIG["seed"]

algorithm_dict = {"PPO": PPO, "DQN": DQN}

if CONFIG["algorithm"] not in algorithm_dict:
    raise ValueError("Invalid algorithm name")

AlgorithmRL = algorithm_dict[CONFIG["algorithm"]]
vae = None
if CONFIG["vae_model"]:
    vae = load_vae(f'./vae/log_dir/{CONFIG["vae_model"]}', LSIZE)
observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(vae, CONFIG["state"])

env = CarlaRouteEnv(obs_res=CONFIG["obs_res"],
                    reward_fn=reward_functions[CONFIG["reward_fn"]],
                    observation_space=observation_space,
                    encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
                    fps=15, action_smoothing=CONFIG["action_smoothing"],
                    action_space_type='continuous', activate_spectator=False)

if reload_model == "":
    model = AlgorithmRL('MultiInputPolicy', env, verbose=1, seed=seed, tensorboard_log=log_dir, device='cpu',
                        **CONFIG["algorithm_params"])
    model_suffix = f"{int(time.time())}"
else:
    model = AlgorithmRL.load(reload_model, env=env, device='cpu', seed=seed, **CONFIG["algorithm_params"])
    model_suffix = f"{reload_model.split('/')[-2].split('_')[-1]}_finetuning"

model_name = f'{model.__class__.__name__}_{model_suffix}'

model_dir = os.path.join(log_dir, model_name)
new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
write_json(CONFIG, os.path.join(model_dir, 'config.json'))

model.learn(total_timesteps=total_timesteps,
            callback=[HParamCallback(CONFIG), TensorboardCallback(1), CheckpointCallback(
                save_freq=total_timesteps // 10,
                save_path=model_dir,
                name_prefix="model")], reset_num_timesteps=False)
