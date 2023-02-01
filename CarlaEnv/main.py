import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import DQN

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from carla_route_env import CarlaRouteEnv
import time
import numpy as np
from wrappers import vector, angle_diff, vector

min_speed = 20.0  # km/h
max_speed = 35.0  # km/h
target_speed = 25.0  # kmh
max_distance = 3.0  # Max distance from center before terminating
low_speed_timer = 0


def reward_fn(env):
    terminal_reason = "Running..."

    # Stop if speed is less than 1.0 km/h after the first 5s of an episode
    global low_speed_timer
    low_speed_timer += 1.0 / env.fps
    speed = env.vehicle.get_speed()
    if low_speed_timer > 5.0 and speed < 1.0:
        env.terminal_state = True
        terminal_reason = "Vehicle stopped"

    # Stop if distance from center > max distance
    #if env.distance_from_center > max_distance:
    #    env.terminal_state = True
    #    terminal_reason = "Off-track"

    # Stop if speed is too high
    if max_speed > 0 and speed > max_speed:
        env.terminal_state = True
        terminal_reason = "Too fast"

    # Calculate reward
    reward = 0
    if not env.terminal_state:
        reward += reward_fn5(env)
    else:
        low_speed_timer = 0.0
        reward -= 10
        print(terminal_reason)


    env.extra_info.extend([
        terminal_reason,
        ""
    ])
    return reward


def reward_fn5(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    # Get angle difference between closest waypoint and vehicle forward vector
    fwd = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle = angle_diff(fwd, wp_fwd)

    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

    # Final reward
    reward = speed_reward * centering_factor * angle_factor + 2 * env.routes_completed

    return reward


"""
log_dir = './tensorboard/'
model = SAC('MlpPolicy', 'Pendulum-v1', tensorboard_log=log_dir, verbose=2).learn(10000, log_interval=1, progress_bar=True)
"""


class HParamCallback(BaseCallback):
    def __init__(self):
        """
        Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
        """
        super().__init__()

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        if self.locals['dones'][0]:
            self.logger.record("custom/total_reward", self.locals['infos'][0]['total_reward'])
            self.logger.record("custom/routes_completed", self.locals['infos'][0]['routes_completed'])
            self.logger.record("custom/total_distance", self.locals['infos'][0]['total_distance'])
            self.logger.record("custom/avg_center_dev", self.locals['infos'][0]['avg_center_dev'])
            self.logger.record("custom/avg_speed", self.locals['infos'][0]['avg_speed'])
            self.logger.dump(self.num_timesteps)
        return True


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

env = CarlaRouteEnv(obs_res=(160, 80), viewer_res=(160 * 7, 80 * 7),
                    reward_fn=reward_fn, start_carla=True, fps=15, action_smoothing=0.7,
                    action_space_type='continuous', activate_spectator=True)

# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
# model = DDPG('CnnPolicy', env, verbose=1, action_noise=action_noise, buffer_size=10000, tensorboard_log=log_dir, device='cpu')
model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir, device='cpu', **ppo_hyperparam)
# model = DQN('MultiInputPolicy', env, verbose=1, tensorboard_log=log_dir, device='cpu', **dqn_hyperparam)
new_logger = configure(log_dir + f'{model.__class__.__name__}_{time.time()}', ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
model.learn(total_timesteps=500_000, callback=[HParamCallback(), TensorboardCallback(1)])
