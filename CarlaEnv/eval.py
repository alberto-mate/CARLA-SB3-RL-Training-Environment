from stable_baselines3 import PPO, DQN
import pandas as pd

from utils import VideoRecorder
from CarlaEnv.vae_commons import create_encode_state_fn, load_vae
from CarlaEnv.rewards import reward_functions

from config import CONFIG

from vae.utils.misc import LSIZE
from CarlaEnv.envs.carla_route_vae_env import CarlaRouteEnv


def run_eval(env, model, model_path=None, record_video=False):
    video_path = model_path.replace(".zip", "_eval.avi")
    csv_path = model_path.replace(".zip", "_eval.csv")
    model_id = model_path.split("/")[-2]
    # vec_env = model.get_env()
    state = env.reset()
    rendered_frame = env.render(mode="rgb_array")

    columns = ["model_id", "episode", "step", "throttle", "steer", "vehicle_location_x", "vehicle_location_y",
                 "reward", "distance", "speed", "center_dev", "angle_next_waypoint"]
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
        print(reward,env.total_reward)
        if dones:
            state = env.reset()

        # Add frame
        rendered_frame = env.render(mode="rgb_array")
        if record_video:
            video_recorder.add_frame(rendered_frame)

        new_row = pd.DataFrame(
            [[model_id, env.episode_idx, env.step_count, env.vehicle.control.steer, env.vehicle.control.throttle,
              env.vehicle.get_transform().location.x, env.vehicle.get_transform().location.y, reward,
              env.distance_traveled,
              env.vehicle.get_speed(), env.distance_from_center, env.vehicle.get_angle(env.current_waypoint)]], columns=columns)
        df = pd.concat([df, new_row], ignore_index=True)

    # Release video
    if record_video:
        video_recorder.release()

    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    model_path = "./tensorboard/PPO_vae64_1677073048/rl_model_50000_steps.zip"

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
