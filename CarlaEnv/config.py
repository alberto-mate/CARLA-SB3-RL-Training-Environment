
import torch as th

_CONFIG_PPO = {
    "algorithm": "PPO",
    "algorithm_params": dict(
        learning_rate=0.0003,
        gae_lambda=0.99,
        ent_coef=0.01,
        n_epochs=5,
        n_steps=1024,
        policy_kwargs=dict(activation_fn=th.nn.ReLU,
                           net_arch=[dict(pi=[500, 300], vf=[500, 300])])
    ),
    "action_smoothing": 0.75,
    "reward": "reward_fn5",
    "reward_params": dict(),
    "obs_res": (160, 80),
}

# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
# model = DDPG('CnnPolicy', env, verbose=1, action_noise=action_noise, buffer_size=10000, tensorboard_log=log_dir, device='cpu')
# dqn_hyperparam = dict(
#     batch_size=100,
#     buffer_size=20_000,
#     target_update_interval=48,
#     exploration_fraction=0.3,
#     learning_starts=5_000,
#     train_freq=4
# )
CONFIG = _CONFIG_PPO
