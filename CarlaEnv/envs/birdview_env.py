import os
import shutil
import subprocess

import carla
import gym
import pygame
from PIL import Image
from pygame.locals import *

from CarlaEnv.hud import HUD
from CarlaEnv.wrappers import *
import time

class CarlaBirdView:
    """
        To be able to drive in this environment, either start start CARLA beforehand with:

        Synchronous:  $> ./CarlaUE4.sh Town07 -benchmark -fps=30
        Asynchronous: $> ./CarlaUE4.sh Town07

        Or pass argument -start_carla in the command-line.
        Note that ${CARLA_ROOT} needs to be set to CARLA's top-level directory
        in order for this option to work.
    """

    def __init__(self, host="127.0.0.1", port=2000, 
                 viewer_res=(1280, 720), fps=15, start_carla=True):
        """
            Initializes an environment that can be used to save camera/sensor data
            from driving around manually in CARLA.

            Connects to a running CARLA enviromment (tested on version 0.9.5) and
            spwans a lincoln mkz2017 passenger car with automatic transmission.

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
            num_images_to_save (int):
                Number of images to collect
            output_dir (str):
                Output directory to save the images to
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

        # Start CARLA from CARLA_ROOT
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
            #launch_command += ['-RenderOffScreen']
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
        self.fps = fps

        self.done = False
        self.recording = False
        self.extra_info = []
        self.num_saved_observations = 0

        self.viewer_image = self.viewer_image_buffer = None                   # Last received image to show in the viewer

        #os.makedirs(os.path.join(self.output_dir, "rgb"))
        #os.makedirs(os.path.join(self.output_dir, "segmentation"))



        self.autopilot = True
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

            # Get spawn location
            lap_start_wp = self.world.map.get_waypoint(carla.Location(x=-180.0, y=110))
            spawn_transform = lap_start_wp.transform
            spawn_transform.location += carla.Location(z=1.0)

            self.spectator = self.world.get_spectator()
            self.spectator.set_transform(sensor_transforms["birdview"])

            # Create hud
            self.hud = HUD(width, height)
            self.world.on_tick(self.hud.on_world_tick)

            # Create cameras
            self.camera  = Camera(self.world, width, height,
                                  transform=sensor_transforms["birdview"],
                                  attach_to=None, on_recv_image=lambda e: self._set_viewer_image(e))




        except Exception as e:
            self.close()
            raise e

        self.hud.notification("Press \"Enter\" to start collecting data.")

    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self):
        # Blit image from spectator camera
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))
        # self.world.debug.draw_point(self.current_waypoint.transform.location + carla.Location(z=1.25), size=0.1,color=carla.Color(0, 255, 255), life_time=2.0, persistent_lines=False)

        for i, point in enumerate(self.world.map.get_spawn_points()):
            self.world.debug.draw_point(point.location  + carla.Location(z=1.25), size=0.02,color=carla.Color(255, 0, 0), life_time=2.0, persistent_lines=False)
            self.world.debug.draw_string(point.location  + carla.Location(x = 3, y=2,z=1.25), str(i),color=carla.Color(0, 0, 255), life_time=2.0)
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = [] # Reset extra info list

        # Render to screen
        pygame.display.flip()

    def step(self):
        if self.is_done():
            raise Exception("Step called after CarlaDataCollector was done.")


        
        # Tick game
        self.world.tick()
        self.clock.tick()
        self.hud.tick(self.world, self.clock)


        # Get most recent observation and viewer image
        self.viewer_image = self._get_viewer_image()

        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            self.done = True
        if keys[K_SPACE]:
            self.recording = True

    def is_done(self):
        return self.done


    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image


    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description="Run this script to drive around with WASD/arrow keys. " +
                                                    "Press SPACE to start recording RGB and semanting segmentation images from the front facing camera to the disk")
    argparser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
    argparser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
    argparser.add_argument("--viewer_res", default="1280x720", type=str, help="Window resolution (default: 1280x720)")
    argparser.add_argument("--obs_res", default="160x80", type=str, help="Output resolution (default: same as --res)")
    argparser.add_argument("--output_dir", default="images", type=str, help="Directory to save images to")
    argparser.add_argument("--num_images", default=12000, type=int, help="Number of images to collect")
    argparser.add_argument("--fps", default=20, type=int, help="FPS. Delta time between samples is 1/FPS")
    argparser.add_argument("--synchronous", type=int, default=True, help="Set this to True when running in a synchronous environment")
    args = argparser.parse_args()

    # Remove existing output directory
    # if os.path.isdir(args.output_dir):
        #shutil.rmtree(args.output_dir)
    #os.makedirs(args.output_dir)

    # Parse viewer_res and obs_res
    viewer_res = [int(x) for x in args.viewer_res.split("x")]
    if args.obs_res is None:
        obs_res = viewer_res
    else:
        obs_res = [int(x) for x in args.obs_res.split("x")]

    # Create vehicle and actors for data collecting
    env = CarlaBirdView(host=args.host, port=args.port,
                                        viewer_res=viewer_res, start_carla=True)

    # While there are more images to collect
    while not env.is_done():
        # Process keyboard input
        pygame.event.pump()
        keys = pygame.key.get_pressed()


        # Take action
        env.step()
        env.render()
        
    # Destroy carla actors
    env.close()
