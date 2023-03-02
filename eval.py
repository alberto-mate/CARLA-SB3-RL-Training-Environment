import os.path

from stable_baselines3 import PPO, DQN
import pandas as pd
import numpy as np

from utils import VideoRecorder
from carla_env.state_commons import create_encode_state_fn, load_vae
from carla_env.rewards import reward_functions

from config import CONFIG

from vae.utils.misc import LSIZE
from carla_env.wrappers import vector, get_displacement_vector
from carla_env.envs.carla_route_env import CarlaRouteEnv
from eval_plots import plot_eval


def run_eval(env, model, model_path=None, record_video=False):
    model_name = os.path.basename(model_path)
    log_path = os.path.join(os.path.dirname(model_path), 'eval')
    os.makedirs(log_path, exist_ok=True)
    video_path = os.path.join(log_path, model_name.replace(".zip", "_eval.avi"))
    csv_path = os.path.join(log_path, model_name.replace(".zip", "_eval.csv"))
    model_id = f"{model_path.split('/')[-2]}-{model_name.split('_')[-2]}"
    # vec_env = model.get_env()
    state = env.reset()
    rendered_frame = env.render(mode="rgb_array")

    columns = ["model_id", "episode", "step", "throttle", "steer", "vehicle_location_x", "vehicle_location_y",
               "reward", "distance", "speed", "center_dev", "angle_next_waypoint", "waypoint_x", "waypoint_y", "route_x", "route_y"]
    df = pd.DataFrame(columns=columns)

    # Init video recording
    if record_video:
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_path, *rendered_frame.shape,
                                                              int(env.fps)))
        video_recorder = VideoRecorder(video_path,
                                       frame_size=rendered_frame.shape,
                                       fps=env.fps)
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    env.episode_idx = 0
    # While non-terminal state
    while env.episode_idx < 2:
        env.extra_info.append("Eval")
        env.extra_info.append("")

        action, _states = model.predict(state, deterministic=True)
        state, reward, dones, info = env.step(action)

        if env.step_count == 2:
            initial_heading = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
            initial_vehicle_location = vector(env.vehicle.get_location())
            # Save the route to plot them later
            for way in env.route_waypoints:
                route_relative = get_displacement_vector(initial_vehicle_location,
                                                         vector(way[0].transform.location),
                                                         initial_heading)
                new_row = pd.DataFrame([['route', env.episode_idx, route_relative[0], route_relative[1]]],
                                       columns=["model_id", "episode", "route_x", "route_y"])
                df = pd.concat([df, new_row], ignore_index=True)

        vehicle_relative = get_displacement_vector(initial_vehicle_location, vector(env.vehicle.get_location()),
                                                   initial_heading)
        waypoint_relative = get_displacement_vector(initial_vehicle_location,
                                                    vector(env.current_waypoint.transform.location), initial_heading)

        new_row = pd.DataFrame(
            [[model_id, env.episode_idx, env.step_count, env.vehicle.control.throttle, env.vehicle.control.steer,
              vehicle_relative[0], vehicle_relative[1], reward,
              env.distance_traveled,
              env.vehicle.get_speed(), env.distance_from_center,
              np.rad2deg(env.vehicle.get_angle(env.current_waypoint)),
              waypoint_relative[0], waypoint_relative[1], None, None
              ]], columns=columns)
        df = pd.concat([df, new_row], ignore_index=True)

        # Add frame
        rendered_frame = env.render(mode="rgb_array")
        if record_video:
            video_recorder.add_frame(rendered_frame)
        if dones:
            state = env.reset()

    # Release video
    if record_video:
        video_recorder.release()

    df.to_csv(csv_path, index=False)
    plot_eval([csv_path])


if __name__ == "__main__":
    model_path = "tensorboard/PPO_vae64_1677524104/model_1400000_steps.zip"

    algorithm_dict = {"PPO": PPO, "DQN": DQN}

    if CONFIG["algorithm"] not in algorithm_dict:
        raise ValueError("Invalid algorithm name")

    vae = load_vae(f'./vae/log_dir/{CONFIG["vae_model"]}', LSIZE)
    observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(vae, CONFIG["state"])

    env = CarlaRouteEnv(obs_res=CONFIG["obs_res"],
                        reward_fn=reward_functions[CONFIG["reward_fn"]],
                        observation_space=observation_space,
                        encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
                        fps=15, action_smoothing=CONFIG["action_smoothing"],
                        action_space_type='continuous', activate_spectator=True, eval=True)

    # model = PPO('MultiInputPolicy', env, verbose=1, device='cpu', **ppo_hyperparam)
    model = PPO.load(model_path, env=env, device='cpu')

    run_eval(env, model, model_path, record_video=True)
