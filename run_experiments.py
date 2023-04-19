import subprocess
import time
import os

experiments = [
    ("1", 200_000),
    ("2", 200_000),
    ("3", 200_000),
    ("4", 500_000),
    ("5", 500_000),
    ("6", 500_000),
]

root_dir = 'tensorboard'
os.environ["CARLA_ROOT"] = "/home/amate/CARLA_0.9.13"
os.environ["SDL_VIDEODRIVER"] = "dummy"

def kill_carla_server():
    # Run killall -9 CarlaUE4-Linux-Shipping
    print("Killing Carla server\n")
    time.sleep(1)
    subprocess.run(["killall", "-9", "CarlaUE4-Linux-Shipping"])
    time.sleep(4)


def get_last_model_path():
    # Get the last model path based after sorting
    dirs = os.listdir(root_dir)
    temp_path = os.path.join(root_dir, sorted(dirs)[-1])
    dirs = os.listdir(temp_path)
    # Check all the .zip files and get the last one
    zip_files = [f for f in dirs if f.endswith('.zip')]
    return os.path.join(temp_path, sorted(zip_files)[-1])


for config, steps in experiments:
    # Check if that config exists already
    for folder in os.listdir(root_dir):
        if "id" in folder and folder.split("id")[1] == config:
            continue

    # Run training
    print(f"Running experiment {config} with {steps} steps\n")
    args_train = [
        "--config", config,
        "--total_timesteps", str(steps),
        "--no_render"
    ]
    subprocess.run(["python", "train.py"] + args_train)
    kill_carla_server()
    # Run evaluation
    print(f"Evaluating experiment {config} with {steps} steps\n")
    print(f"model path: {get_last_model_path()}")
    last_model_path = get_last_model_path()
    args_eval = [
        "--config", config,
        "--model", last_model_path,
    ]

    subprocess.run(["python", "eval.py"] + args_eval)
