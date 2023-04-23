import subprocess
import time
import os

experiments = [
    ("3", 2_000),
]

root_dir = 'tensorboard'
os.environ["CARLA_ROOT"] = "/home/albertomate/Documentos/carla"
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
    max_steps = 0
    latest_model = ''
    for file in dirs:
        if file.endswith('.zip') and file.startswith('model_'):
            steps = int(file.split('_')[1].split('.')[0])
            if steps > max_steps:
                max_steps = steps
                latest_model = file

    return os.path.join(temp_path, latest_model)


def main():
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


if __name__ == "__main__":
    main()
