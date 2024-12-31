"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect demonstrations for Isaac Lab environments."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment",
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=1,
    help="Number of episodes to store in the dataset.",
)
parser.add_argument(
    "--filename", type=str, default="hdf_dataset", help="Basename of output file."
)
# parser.add_argument("--active_gpu", type=int, default=2)
# parser.add_argument("--physics_gpu", type=int, default=0)
parser.add_argument("--multi_gpu", action="store_false", default=True)
parser.add_argument("--max_gpu_count", type=int, default=3)
# append AppLauncher cli args

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.core.utils.extensions import enable_extension

EXTENSIONS = [
    "omni.anim.skelJoint",
]

for ext in EXTENSIONS:
    enable_extension(ext)

"""Rest everything follows."""

import torch
import os
import time

from yaml_config_utils import get_config, load_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from utils.utils import set_lmp_objects
import engine_interfaces
from envs.isaac_env.isaaclab_panda_env import EnvIsaacLab

if __name__ == "__main__":
    config_path = "/home/ps/Projects/isaac-lab-workspace/IsaacLab_1.3?/Voxposer/src/configs/isaac_panda.yaml"
    env_config = get_config(config_path=config_path)
    log_dir = os.path.join(
        env_config.log_dir,
        f"{time.strftime('%Y-%m-%d-%H-%M-%S')}",
    )
    env_config["visualizer"]["save_dir"] = os.path.join(log_dir, "visualizer")
    scene_target_objects = env_config.env["scene_target_objects"]
    tgi_config = load_config(
        "/home/ps/Projects/isaac-lab-workspace/IsaacLab_1.3?/Voxposer/src/configs/TGI_deepseek-coder-33B-instruct-AWQ.yaml"
    )
    tgi = getattr(engine_interfaces, tgi_config["type"])(**tgi_config)
    visualizer = ValueMapVisualizer(env_config["visualizer"])
    env = EnvIsaacLab(
        task_name=args_cli.task, cfg=env_config["env"], visualizer=visualizer
    )
    input("waiting for start!")
    obs = env.reset()
    descriptions = env_config.env["description"]
    lmps, lmp_env = setup_LMP(env, env_config, debug=False, engine_call_fn=tgi)
    voxposer_ui = lmps["plan_ui"]
    set_lmp_objects(lmps, env.get_object_names())
    voxposer_ui(descriptions)
