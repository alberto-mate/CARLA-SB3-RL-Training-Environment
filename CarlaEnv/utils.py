import types

import cv2
import numpy as np
import scipy.signal
import tensorflow as tf
import math


class VideoRecorder():
    def __init__(self, filename, frame_size, fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(filename,fourcc, int(fps),(frame_size[1], frame_size[0]))

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()


from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class HParamCallback(BaseCallback):
    def __init__(self, config):
        """
        Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
        """
        super().__init__()
        self.config = config

    def _on_training_start(self) -> None:
        hparam_dict = {}
        for k, v in self.config.items():
            if isinstance(v, str) and v.isnumeric():
                hparam_dict[k] = int(v)
            else:
                hparam_dict[k] = v.__str__()
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
            self.logger.record("custom/mean_reward", self.locals['infos'][0]['mean_reward'])
            self.logger.dump(self.num_timesteps)
        return True


def lr_schedule(initial_value: float, end_value: float, rate: float):
    """
    Learning rate schedule:
        Exponential decay by factors of 10 from initial_value to end_value.

    :param initial_value: Initial learning rate.
    :param rate: Exponential rate of decay. High values mean fast early drop in LR
    :param end_value: The final value of the learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: A float value between 0 and 1 that represents the remaining progress.
        :return: The current learning rate.
        """
        if progress_remaining <= 0:
            return end_value

        return end_value + (initial_value - end_value) * (10 ** (rate * math.log10(progress_remaining)))

    return func