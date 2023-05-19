import torch as th
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
from utils import lr_schedule

algorithm_params = {
    "PPO": dict(
        learning_rate=lr_schedule(1e-4, 1e-6, 2),
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=dict(activation_fn=th.nn.ReLU,
                           net_arch=[dict(pi=[500, 300], vf=[500, 300])])
    ),
    "SAC": dict(
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]),
    ),
    "DDPG": dict(
        gamma=0.98,
        buffer_size=200000,
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        gradient_steps=-1,
        learning_rate=lr_schedule(5e-4, 1e-6, 2),
        policy_kwargs=dict(net_arch=[400, 300]),
    ),
    "SAC_BEST": dict(
        learning_rate=lr_schedule(1e-4, 5e-7, 2),
        buffer_size=300000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.98,
        tau=0.02,
        train_freq=64,
        gradient_steps=64,
        learning_starts=10000,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3, net_arch=[500, 300]),
    ),
}

states = {
    "1": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver"],
    "2": ["steer", "throttle", "speed", "maneuver"],
    "3": ["steer", "throttle", "speed", "waypoints"],
    "4": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver", "distance_goal"]
}

reward_params = {
    "reward_fn_5_default": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,  # Max distance from center before terminating
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
     "reward_fn_5_no_early_stop": dict(
         early_stop=False,
         min_speed=20.0,  # km/h
         max_speed=35.0,  # km/h
         target_speed=25.0,  # kmh
         max_distance=3.0,  # Max distance from center before terminating
         max_std_center_lane=0.4,
         max_angle_center_lane=90,
         penalty_reward=-10,
     ),
    "reward_fn_5_best": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=2.0,  # Max distance from center before terminating
        max_std_center_lane=0.35,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
}

_CONFIG_1 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["3"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_2 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["3"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_3 = {
    "algorithm": "DDPG",
    "algorithm_params": algorithm_params["DDPG"],
    "state": states["3"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_4 = {
    "algorithm": "PPO",
    "algorithm_params": algorithm_params["PPO"],
    "state": states["1"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_5 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_6 = {
    "algorithm": "DDPG",
    "algorithm_params": algorithm_params["DDPG"],
    "state": states["1"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_7 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_8 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64",
    "action_smoothing": 0,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_9 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_no_early_stop"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_10 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["2"],
    "vae_model": "vae_64", # Cambiar mejor VAE
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_11 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["4"],
    "vae_model": "vae_64", # Cambiar mejor VAE
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_12 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}

_CONFIG_13 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": ["HistoryWrapperObsDict_5"]
}

_CONFIG_14 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": ["FrameSkip_3"]
}

_CONFIG_15 = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC"],
    "state": states["1"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_default"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": ["FrameSkip_3", "HistoryWrapperObsDict_5"]
}

_CONFIG_BEST = {
    "algorithm": "SAC",
    "algorithm_params": algorithm_params["SAC_BEST"],
    "state": states["1"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": reward_params["reward_fn_5_best"],
    "obs_res": (160, 80),
    "seed": 100,
    "wrappers": []
}
CONFIGS = {
    "1": _CONFIG_1,
    "2": _CONFIG_2,
    "3": _CONFIG_3,
    "4": _CONFIG_4,
    "5": _CONFIG_5,
    "6": _CONFIG_6,
    "7": _CONFIG_7,
    "8": _CONFIG_8,
    "9": _CONFIG_9,
    "10": _CONFIG_10,
    "11": _CONFIG_11,
    "12": _CONFIG_12,
    "13": _CONFIG_13,
    "14": _CONFIG_14,
    "15": _CONFIG_15,
    "BEST": _CONFIG_BEST
}
CONFIG = None


def set_config(config_name):
    global CONFIG
    CONFIG = CONFIGS[config_name]
