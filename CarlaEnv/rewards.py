from wrappers import angle_diff, vector
import numpy as np
from config import CONFIG

#min_speed = 20.0  # km/h
#max_speed = 35.0  # km/h
#target_speed = 25.0  # kmh
#max_distance = 3.0  # Max distance from center before terminating
low_speed_timer = 0

min_speed = CONFIG["reward_params"]["min_speed"]
max_speed = CONFIG["reward_params"]["max_speed"]
target_speed = CONFIG["reward_params"]["target_speed"]
max_distance = CONFIG["reward_params"]["max_distance"]
reward_functions = {}
def create_reward_fn(reward_fn):
    def func(env):
        terminal_reason = "Running..."

        # Stop if speed is less than 1.0 km/h after the first 5s of an episode
        global low_speed_timer
        low_speed_timer += 1.0 / env.fps
        speed = env.vehicle.get_speed()
        if low_speed_timer > 5.0 and speed < 1.0 and env.current_waypoint_index >= 1:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        # Stop if distance from center > max distance
        if env.distance_from_center > max_distance:
            env.terminal_state = True
            terminal_reason = "Off-track"

        # Stop if speed is too high
        if max_speed > 0 and speed > max_speed:
            env.terminal_state = True
            terminal_reason = "Too fast"

        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        else:
            low_speed_timer = 0.0
            reward -= 10
            print(terminal_reason)


        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward
    return func

# Reward_fn5
def reward_fn5(env):
    """
        reward = Positive speed reward for being close to target speed,
                 however, quick decline in reward beyond target speed
               * centering factor (1 when centered, 0 when not)
               * angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    """

    # Get angle difference between closest waypoint and vehicle forward vector
    # fwd = vector(env.vehicle.get_velocity())
    # wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    # angle = angle_diff(fwd, wp_fwd)

    angle = env.vehicle.get_angle(env.current_waypoint)
    speed_kmh = env.vehicle.get_speed()
    if speed_kmh < min_speed:  # When speed is in [0, min_speed] range
        speed_reward = speed_kmh / min_speed  # Linearly interpolate [0, 1] over [0, min_speed]
    elif speed_kmh > target_speed:  # When speed is in [target_speed, inf]
        # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:  # Otherwise
        speed_reward = 1.0  # Return 1 for speeds in range [min_speed, target_speed]

    # Interpolated from 1 when centered to 0 when 3 m from center
    centering_factor = max(1.0 - (env.distance_from_center - 0.22) / max_distance, 0.0)

    # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
    angle_factor = max(1.0 - abs(angle / np.deg2rad(90)), 0.0)

    std = np.std(env.distance_from_center_history)
    distance_factor = max(1.0 - abs(std / 0.6), 0.0)

    # Final reward
    reward = speed_reward * centering_factor * angle_factor * distance_factor

    return reward

reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)
