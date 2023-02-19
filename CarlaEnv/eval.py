
from stable_baselines3 import PPO

from utils import VideoRecorder, compute_gae
from CarlaEnv.vae_commons import create_encode_state_fn, load_vae
from CarlaEnv.rewards import reward_fn


from vae.utils.misc import LSIZE
from CarlaEnv.envs.carla_route_vae_env import CarlaRouteEnv as CarlaEnv



def run_eval(env, model, video_filename=None):
    # Init test env
    terminal = False
    total_reward = 0
    vec_env = model.get_env()
    state = vec_env.reset()
    rendered_frame = env.render(mode="rgb_array")

    # Init video recording
    if video_filename is not None:
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
                                                              int(env.average_fps)))
        video_recorder = VideoRecorder(video_filename,
                                       frame_size=rendered_frame.shape,
                                       fps=env.average_fps)
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    episode_idx = 0

    # While non-terminal state
    while not terminal:
        env.extra_info.append("Episode {}".format(episode_idx))
        env.extra_info.append("Running eval...".format(episode_idx))
        env.extra_info.append("")

        action, _states = model.predict(state, deterministic=True)
        state, reward, dones, info = vec_env.step(action)

        if dones:
            episode_idx+=1

        terminal = env.terminal_state

        # Add frame
        rendered_frame = vec_env.render(mode="rgb_array")
        if video_recorder is not None:
            video_recorder.add_frame(rendered_frame)
        total_reward += reward

    # Release video
    if video_recorder is not None:
        video_recorder.release()



    return total_reward


if __name__ == "__main__":

    vae = load_vae(f'/home/albertomate/Documentos/carla/PythonAPI/my-carla/vae/log_dir/vae_{LSIZE}_12000data', LSIZE)
    encode_state_fn, decode_vae_fn = create_encode_state_fn(vae)

    env = CarlaEnv(obs_res=(160, 80), viewer_res=(160 * 7, 80 * 7),
                                reward_fn=reward_fn, encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
                                start_carla=True, fps=15, action_smoothing=0.7,
                                action_space_type='continuous', activate_spectator=True)

    # model = PPO('MultiInputPolicy', env, verbose=1, device='cpu', **ppo_hyperparam)
    model = PPO.load(
        "/home/albertomate/Documentos/carla/PythonAPI/my-carla/CarlaEnv/tensorboard/PPO_VAE64_1675553190.3264425/PPO_VAE64_1675553190.3264425_1500000",
        env=env, device='cpu')

    run_eval(env, model)