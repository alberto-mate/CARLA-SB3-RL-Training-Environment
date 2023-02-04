from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from CarlaEnv.envs.carla_route_vae_env import CarlaRouteEnv
import time
from vae.utils.misc import LSIZE
from vae_commons import create_encode_state_fn, load_vae

from rewards import reward_fn
from callbacks import HParamCallback, TensorboardCallback



log_dir = './tensorboard/'

dqn_hyperparam = dict(
    batch_size=100,
    buffer_size=20_000,
    target_update_interval=48,
    exploration_fraction=0.3,
    learning_starts=5_000,
    train_freq=4
)

ppo_hyperparam = dict(
    learning_rate=0.0001,
    gae_lambda=0.99,
    ent_coef=0.01,
    n_epochs=5,
)

vae = load_vae(f'/home/albertomate/Documentos/carla/PythonAPI/my-carla/vae/log_dir/vae_{LSIZE}',LSIZE)
encode_state_fn = create_encode_state_fn(vae)

env = CarlaRouteEnv(obs_res=(160, 80), viewer_res=(160 * 7, 80 * 7),
                    reward_fn=reward_fn, encode_state_fn=encode_state_fn,
                    start_carla=True, fps=15, action_smoothing=0.7,
                    action_space_type='continuous', activate_spectator=False)

# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
# model = DDPG('CnnPolicy', env, verbose=1, action_noise=action_noise, buffer_size=10000, tensorboard_log=log_dir, device='cpu')

model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir, device='cpu', **ppo_hyperparam)
new_logger = configure(log_dir + f'{model.__class__.__name__}_{time.time()}', ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
model.learn(total_timesteps=500_000, callback=[HParamCallback(), TensorboardCallback(1)])
