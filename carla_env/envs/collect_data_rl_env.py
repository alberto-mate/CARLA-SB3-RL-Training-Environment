import os
import subprocess
import sys
import glob

import gym
import pygame
from gym.utils import seeding
from pygame.locals import *
from PIL import Image

from carla_env.tools.hud import HUD
from carla_env.navigation.planner import compute_route_waypoints
from carla_env.wrappers import *
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla




class CarlaCollectDataRLEnv(gym.Env):
    """
        This is a simple CARLA environment where the goal is to drive in a lap
        around the outskirts of Town07. This environment can be used to compare
        different models/reward functions in a realtively predictable environment.

        To run an agent in this environment, either start start CARLA beforehand with:

        Synchronous:  $> ./CarlaUE4.sh Town07 -benchmark -fps=30
        Asynchronous: $> ./CarlaUE4.sh Town07

        Or, pass argument -start_carla in the command-line.
        Note that ${CARLA_ROOT} needs to be set to CARLA's top-level directory
        in order for this option to work.

        And also remember to set the -fps and -synchronous arguments to match the
        command-line arguments of the simulator (not needed with -start_carla.)

        Note that you may also need to add the following line to
        Unreal/CarlaUE4/Config/DefaultGame.ini to have the map included in the package:

        +MapsToCook=(FilePath="/Game/Carla/Maps/Town07")
    """

    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, host="127.0.0.1", port=2000,
                 viewer_res=(1280, 720), obs_res=(1280, 720), observation_space=None,
                 reward_fn=None, encode_state_fn=None, decode_vae_fn = None,
                 num_images_to_save=20000, output_dir="images",
                 fps=15, action_smoothing=0.0, action_space_type="continuous",
                 activate_spectator=True,
                 start_carla=True):
        """
            Initializes a gym-like environment that can be used to interact with CARLA.

            Connects to a running CARLA enviromment (tested on version 0.9.5) and
            spwans a lincoln mkz2017 passenger car with automatic transmission.

            This vehicle can be controlled using the step() function,
            taking an action that consists of [steering_angle, throttle].

            host (string):
                IP address of the CARLA host
            port (short):
                Port used to connect to CARLA
            viewer_res (int, int):
                Resolution of the spectator camera (placed behind the vehicle by default)
                as a (width, height) tuple
            obs_res (int, int):
                Resolution of the observation camera (placed on the dashboard by default)
                as a (width, height) tuple
            reward_fn (function):
                Custom reward function that is called every step.
                If None, no reward function is used.
            encode_state_fn (function):
                Function that takes the image (of obs_res resolution) from the
                observation camera and encodes it to some state vector to returned
                by step(). If None, step() returns the full image.
            action_smoothing:
                Scalar used to smooth the incomming action signal.
                1.0 = max smoothing, 0.0 = no smoothing
            fps (int):
                FPS of the client. If fps <= 0 then use unbounded FPS.
                Note: Sensors will have a tick rate of fps when fps > 0,
                otherwise they will tick as fast as possible.
            synchronous (bool):
                If True, run in synchronous mode (read the comment above for more info)
            start_carla (bool):
                Automatically start CALRA when True. Note that you need to
                set the environment variable ${CARLA_ROOT} to point to
                the CARLA root directory for this option to work.
        """

        self.carla_process = None
        if start_carla:
            if "CARLA_ROOT" not in os.environ:
                raise Exception("${CARLA_ROOT} has not been set!")
            carla_path = os.environ["CARLA_ROOT"]
            carla_path = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4.sh")
            launch_command = [carla_path]
            launch_command += ['-quality_level=Low']
            launch_command += ['-benchmark']
            launch_command += ["-fps=%i" % fps]
            launch_command += ['-RenderOffScreen']
            launch_command += ['-prefernvidia']
            # launch_command += ['-dx11']
            print("Running command:")
            print(" ".join(launch_command))
            self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.DEVNULL)
            print("Waiting for CARLA to initialize")

            # ./CarlaUE4.sh -quality_level=Low -benchmark -fps=15 -RenderOffScreen
            time.sleep(5)

        # Initialize pygame for visualization
        pygame.init()
        pygame.font.init()
        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        # Setup gym environment
        self.seed()
        self.action_space_type = action_space_type
        if self.action_space_type == "continuous":
            self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)  # steer, throttle
        elif self.action_space_type == "discrete":
            self.action_space = gym.spaces.Discrete(4)

        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(out_height, out_width, 3), dtype=np.uint8)
        self.observation_space = observation_space

        self.fps = fps
        self.action_smoothing = action_smoothing



        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.decode_vae_fn = (lambda x: x) if not callable(decode_vae_fn) else decode_vae_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.max_distance = 3000  # m
        self.activate_spectator = activate_spectator

        self.output_dir = output_dir
        self.done = False
        self.recording = False
        self.extra_info = []
        self.num_saved_observations = 12_000
        self.num_images_to_save = num_images_to_save
        self.episode_idx = -2
        self.observation = {key: None for key in ["rgb", "segmentation"]}  # Last received observations
        self.observation_buffer = {key: None for key in ["rgb", "segmentation"]}
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer

        self.world = None
        try:
            # Connect to carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(60.0)

            # Create world wrapper
            self.world = World(self.client)

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1 / self.fps
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.client.reload_world(False)  # reload map keeping the world settings

            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0],
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

            # Create hud
            self.hud = HUD(width, height)
            self.hud.set_vehicle(self.vehicle)
            self.world.on_tick(self.hud.on_world_tick)

            # Create cameras

            self.dashcam_rgb = Camera(self.world, out_width, out_height,
                                      transform=sensor_transforms["dashboard"],
                                      attach_to=self.vehicle,
                                      on_recv_image=lambda e: self._set_observation_image("rgb", e))
            self.dashcam_seg = Camera(self.world, out_width, out_height,
                                      transform=sensor_transforms["dashboard"],
                                      attach_to=self.vehicle,
                                      on_recv_image=lambda e: self._set_observation_image("segmentation", e),
                                      camera_type="sensor.camera.semantic_segmentation", custom_palette=True)

            # self.camera = Camera(self.world, width, height,
            #                      transform=sensor_transforms["spectator"],
            #                      attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e))
        except Exception as e:
            self.close()
            raise e
        # Reset env to set initial state
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, is_training=False):
        # Create new route
        self.num_routes_completed = -1
        self.new_route()
        self.distance_from_center_history = []
        self.success_state = False
        self.episode_idx += 1
        # Set env vars
        self.terminal_state = False  # Set to True when we want to end episode
        self.closed = False  # Set to True when ESC is pressed
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = {key: None for key in ["rgb", "segmentation"]}  # Last received observations
        self.observation_buffer = {key: None for key in ["rgb", "segmentation"]}
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.step_count = 0

        # Init metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0
        self.world.tick()
        # Return initial observation
        time.sleep(0.4)
        obs = self.step(None)[0]
        time.sleep(0.2)
        return obs

    def new_route(self):
        # Do a soft reset (teleport vehicle)
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        # self.vehicle.control.brake = float(0.0)
        self.vehicle.tick()
        self.vehicle.set_simulate_physics(False)  # Reset the car's physics

        # Generate waypoints along the lap
        self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in
                                      np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)]

        self.route_waypoints = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)
        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.vehicle.set_transform(self.start_wp.transform)
        self.vehicle.set_simulate_physics(True)

    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):

        # Blit image from spectator camera
        # self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Superimpose current observation into top-right corner
        for i, (_, obs) in enumerate(self.observation.items()):
            obs_h, obs_w = obs.shape[:2]
            # view_h, view_w = self.viewer_image.shape[:2]
            view_w = 160*7
            pos = (view_w - obs_w - 10, obs_h * i + 10 * (i + 1))
            self.display.blit(pygame.surfarray.make_surface(obs.swapaxes(0, 1)), pos)

        # Save current observations
        if self.recording and self.vehicle.get_speed() > 2.0 and self.distance_from_center > 0.75:
            for obs_type, obs in self.observation.items():
                img = Image.fromarray(obs)
                img.save(os.path.join(self.output_dir, obs_type, "{}.png".format(self.num_saved_observations)))
            self.num_saved_observations += 1
            if self.num_saved_observations >= self.num_images_to_save:
                self.done = True

        # Render HUDw
        self.extra_info.extend([
            "Images: %i/%i" % (self.num_saved_observations, self.num_images_to_save),
            "Progress: %.2f%%" % (self.num_saved_observations / self.num_images_to_save * 100.0),
            "",
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
        ])
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list

        # Render to screen
        pygame.display.flip()

    def step(self, action):
        if self.is_done():
            raise Exception("Step called after CarlaDataCollector was done.")
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")

        # Create new route on route completion
        if self.current_waypoint_index >= len(self.route_waypoints) - 1:
            self.new_route()

        # Take action
        if action is not None:
            if self.action_space_type == "continuous":
                steer, throttle = [float(a) for a in action]
            elif self.action_space_type == "discrete":
                possible_actions = {
                    0: [-1, 1],
                    1: [0, 1],
                    2: [1, 1],
                    3: [0, 0],

                }
                steer, throttle = possible_actions[action]

            # steer, throttle, brake = [float(a) for a in action]
            self.vehicle.control.steer = self.vehicle.control.steer * self.action_smoothing + steer * (
                    1.0 - self.action_smoothing)
            self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (
                    1.0 - self.action_smoothing)
            # self.vehicle.tick()
            # self.vehicle.control.brake = self.vehicle.control.brake * self.action_smoothing + brake * (1.0-self.action_smoothing)

        # Tick game
        self.world.tick()
        self.clock.tick()
        self.hud.tick(self.world, self.clock)

        # Get most recent observation and viewer image
        self.observation["rgb"] = self._get_observation("rgb")
        self.observation["segmentation"] = self._get_observation("segmentation")

        #self.viewer_image = self._get_viewer_image()


        encoded_state = self.encode_state_fn(self)
        self.observation_decoded = self.decode_vae_fn(encoded_state)

        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Keep track of closest waypoint on the route
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0:  # Did we pass the waypoint?
                waypoint_index += 1  # Go to next waypoint
            else:
                break
        self.current_waypoint_index = waypoint_index

        # Check for route completion
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]

        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
            self.route_waypoints)

        # Calculate deviation from center of the lane
        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        # Calculate distance traveled
        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()

        # Terminal on max distance
        if self.distance_traveled >= self.max_distance:
            self.terminal_state = True

        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward
        self.step_count += 1

        state = {
            'vae_latent': encoded_state,
            'vehicle_measures': np.array(
                [self.vehicle.control.steer, self.vehicle.control.throttle, self.vehicle.get_speed()],
                dtype=np.float32),  # steer, throttle, speed
            'maneuver': self.current_road_maneuver.value,
        }

        # DEBUG: Draw path
        # self._draw_path(life_time=2.0, skip=10)
        # DEBUG: Draw current waypoint
        # self.world.debug.draw_point(self.current_waypoint.transform.location + carla.Location(z=1.25), size=0.1,color=carla.Color(0, 255, 255), life_time=2.0, persistent_lines=False)

        # Check for ESC press
        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            self.close()
            self.terminal_state = True

        if pygame.key.get_pressed()[K_SPACE]:
            self.recording = not self.recording
        self.render()

        info = {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count)
        }
        return state, self.last_reward, self.terminal_state, info

    def is_done(self):
        return self.done

    def _get_observation(self, name):
        while self.observation_buffer[name] is None:
            pass
        obs = self.observation_buffer[name].copy()
        self.observation_buffer[name] = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _on_collision(self, event):
        self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))
        self.terminal_state = True

    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, name, image):
        self.observation_buffer[name] = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image




from stable_baselines3 import PPO
import time
from vae.utils.misc import LSIZE
from carla_env.state_commons import create_encode_state_fn, load_vae

from carla_env.rewards import reward_functions
from config import CONFIG
from utils import HParamCallback, TensorboardCallback, write_json, parse_wrapper_class



ppo_hyperparam = dict(
    learning_rate=0.0003,
    gae_lambda=0.99,
    ent_coef=0.01,
    n_epochs=5,
    n_steps=1024
)


observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(None, ["steer", "throttle", "speed"])


env = CarlaCollectDataRLEnv(obs_res=(160, 80), viewer_res=(160 * 7, 80 * 7),
                    reward_fn=reward_functions['reward_fn5'], encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn, observation_space=observation_space,
                    start_carla=True, fps=10, action_smoothing=0.7,
                    action_space_type='continuous', activate_spectator=True, output_dir='/home/albertomate/Documentos/carla/PythonAPI/my-carla/vae/images2')

model = PPO('MultiInputPolicy', env, device='cpu')
# model = PPO('MultiInputPolicy', env, verbose=1, device='cpu', **ppo_hyperparam)
model_name = f'{model.__class__.__name__}_VAE{LSIZE}_{time.time()}'
model.learn(total_timesteps=500_000, reset_num_timesteps=False)

