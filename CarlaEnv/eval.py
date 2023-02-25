import os.path

from stable_baselines3 import PPO, DQN
import pandas as pd
import numpy as np

from utils import VideoRecorder
from CarlaEnv.vae_commons import create_encode_state_fn, load_vae
from CarlaEnv.rewards import reward_functions

from config import CONFIG

from vae.utils.misc import LSIZE
from wrappers import vector, get_displacement_vector
from CarlaEnv.envs.carla_route_vae_env import CarlaRouteEnv


def run_eval(env, model, model_path=None, record_video=False):
    model_name = os.path.basename(model_path)
    log_path = os.path.join(os.path.dirname(model_path), 'eval')
    os.makedirs(log_path, exist_ok=True)
    video_path = os.path.join(log_path, model_name.replace(".zip", "_eval.avi"))
    csv_path = os.path.join(log_path, model_name.replace(".zip", "_eval.csv"))
    model_id = model_path.split("/")[-2]
    # vec_env = model.get_env()
    state = env.reset()
    rendered_frame = env.render(mode="rgb_array")

    columns = ["model_id", "episode", "step", "throttle", "steer", "vehicle_location_x", "vehicle_location_y",
               "reward", "distance", "speed", "center_dev", "angle_next_waypoint", "waypoint_x", "waypoint_y"]
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

    # While non-terminal state
    while env.episode_idx < 2:
        env.extra_info.append("Eval - Episode {}".format(env.episode_idx))
        env.extra_info.append("")

        action, _states = model.predict(state, deterministic=True)
        state, reward, dones, info = env.step(action)
        print(reward, env.total_reward)

        if env.step_count == 2:
            initial_heading = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
            initial_vehicle_location = vector(env.vehicle.get_location())

        vehicle_relative = get_displacement_vector(initial_vehicle_location, vector(env.vehicle.get_location()),
                                                   initial_heading)
        waypoint_relative = get_displacement_vector(initial_vehicle_location,
                                                    vector(env.current_waypoint.transform.location), initial_heading)
        print(waypoint_relative)

        new_row = pd.DataFrame(
            [[model_id, env.episode_idx, env.step_count, env.vehicle.control.throttle, env.vehicle.control.steer,
              vehicle_relative[0], vehicle_relative[1], reward,
              env.distance_traveled,
              env.vehicle.get_speed(), env.distance_from_center, env.vehicle.get_angle(env.current_waypoint),
              waypoint_relative[0], waypoint_relative[1]
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
    plot_eval(csv_path)


def plot_eval(eval_csv_path):
    import matplotlib.pyplot as plt

    # Load the dataframe
    df = pd.read_csv(eval_csv_path)

    # Get a list of unique episode numbers
    episode_numbers = df['episode'].unique()
    cols = ['Steer', 'Throttle', 'Speed (km/h)', 'Reward', 'Center Deviation (m)', 'Distance (m)',
            'Angle next waypoint (rad)', 'Trajectory']

    # Create a figure with subplots for each episode
    fig, axs = plt.subplots(len(episode_numbers), len(cols), figsize=(4 * len(cols), 3 * len(episode_numbers)))

    # Loop over each episode number
    for i, episode_number in enumerate(episode_numbers):
        # Select the rows for the current episode
        episode_df = df[df['episode'] == episode_number]

        # Plot the steer progress
        axs[i][0].plot(episode_df['step'], episode_df['steer'])
        axs[i][0].set_xlabel('Step')
        axs[i, 0].set_ylim(-1, 1)  # clip y-axis limits to -1 and 1

        # Plot the throttle progress
        axs[i][1].plot(episode_df['step'], episode_df['throttle'])
        axs[i][1].set_xlabel('Step')
        axs[i, 1].set_ylim(0, 1)  # clip y-axis limits to -1 and 1

        axs[i][2].plot(episode_df['step'], episode_df['speed'])
        axs[i][2].set_xlabel('Step')
        axs[i, 2].set_ylim(0, 30)  # clip y-axis limits to -1 and 1

        # Plot the reward progress
        axs[i][3].plot(episode_df['step'], episode_df['reward'])
        axs[i][3].set_xlabel('Step')
        axs[i, 3].set_ylim(-0.2, 1)  # clip y-axis limits to -1 and 1

        axs[i][4].plot(episode_df['step'], episode_df['center_dev'])
        axs[i][4].set_xlabel('Step')
        axs[i, 4].set_ylim(0, 3)  # clip y-axis limits to -1 and 1

        axs[i][5].plot(episode_df['step'], episode_df['distance'])
        axs[i][5].set_xlabel('Step')

        axs[i][6].plot(episode_df['step'], episode_df['angle_next_waypoint'])
        axs[i][6].set_xlabel('Step')

        axs[i][7].plot(episode_df['vehicle_location_x'], episode_df['vehicle_location_y'], label='Vehicle')
        axs[i][7].plot(episode_df['waypoint_x'], episode_df['waypoint_y'], label='Waypoint')
        axs[i][7].plot(episode_df['vehicle_location_x'].head(1), episode_df['vehicle_location_y'].head(1), 'go',
                       label='Start')
        axs[i][7].plot(episode_df['vehicle_location_x'].tail(1), episode_df['vehicle_location_y'].tail(1), 'ro',
                       label='End')

        axs[i, 7].set_xlim(left=min(-2, min(episode_df['waypoint_x'])))
        axs[i, 7].set_xlim(right=max(2, max(episode_df['waypoint_x'])))

    pad = 5  # in points
    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, row in zip(axs[:, 0], episode_numbers):
        ax.annotate(f"Episode {row}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.savefig(os.path.join(os.path.dirname(eval_csv_path), "plot.png"))


if __name__ == "__main__":
    model_path = "./tensorboard/PPO_vae64_1677320181/rl_model_20000_steps.zip"

    algorithm_dict = {"PPO": PPO, "DQN": DQN}

    if CONFIG["algorithm"] not in algorithm_dict:
        raise ValueError("Invalid algorithm name")

    vae = load_vae(f'../vae/log_dir/{CONFIG["vae_model"]}', LSIZE)
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
