import torch as th

from utils import lr_schedule

_CONFIG_PPO = {
    "algorithm": "PPO",
    "algorithm_params": dict(
        learning_rate=lr_schedule(1e-4, 1e-6, 2),
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        n_epochs=10,
        n_steps=1024,
        policy_kwargs=dict(activation_fn=th.nn.ReLU,
                           net_arch=[dict(pi=[500, 300], vf=[500, 300])])
    ),
    "state": ["steer", "throttle", "speed", "angle_next_waypoint", "maneuver", "distance_goal"],
    "vae_model": "vae_64",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn_waypoints",
    "reward_params": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,  # Max distance from center before terminating
        max_std_center_lane=0.4,
        max_angle_center_lane=90,
        penalty_reward=-100,
    ),
    "obs_res": (160, 80),
    "seed": 436,
}

_CONFIG_SAC = {
    "algorithm": "SAC",
    "algorithm_params": dict(
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
    "state": ["steer", "throttle", "speed", "maneuver"],
    "vae_model": "vae_64_augmentation",
    "action_smoothing": 0.75,
    "reward_fn": "reward_fn5",
    "reward_params": dict(
        early_stop=True,
        min_speed=20.0,  # km/h
        max_speed=35.0,  # km/h
        target_speed=25.0,  # kmh
        max_distance=3.0,  # Max distance from center before terminating
        max_std_center_lane=0.5,
        max_angle_center_lane=90,
        penalty_reward=-10,
    ),
    "obs_res": (160, 80),
    "seed": 34435,
    "wrappers": ['HistoryWrapperObsDict_5']
}

CONFIG = _CONFIG_SAC
