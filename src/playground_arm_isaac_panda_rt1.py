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


import requests
import base64
from PIL import Image
import io
import numpy as np

# # 准备测试数据
# image = Image.new("RGB", (640, 480))
# buffered = io.BytesIO()
# image.save(buffered, format="JPEG")
# image_base64 = base64.b64encode(buffered.getvalue()).decode()

# data = {"image": image_base64, "instruction": "Pick up the red block"}

# # 发送请求
# response = requests.post("http://localhost:8000/predict", json=data)
# print(response.json())


def prepare_date(image, instruction="Pick up the cube"):
    """
    将相机图像转换为RT1所需格式并发送请求

    Args:
        image: CHW格式的numpy数组
        instruction: RT1指令字符串
    Returns:
        response: RT1响应数据
    """
    # 转换图像格式
    image_chw = image  # shape: (C,H,W)
    image_hwc = np.transpose(image_chw, (1, 2, 0))  # shape: (H,W,C)
    image_hwc = image_hwc.astype(np.uint8)

    # 转换为PIL Image并编码
    pil_image = Image.fromarray(image_hwc)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    # 准备请求数据
    data = {"image": image_base64, "instruction": instruction}
    return data


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
    env.turn_off_vlm()
    obs = env.reset()
    front_rgb = obs["camera"]["front_rgb"]
    data = prepare_date(front_rgb)
    reset_response = requests.post("http://localhost:8000/reset")
    if reset_response.status_code == 200:
        print("Reset successful!")
    else:
        print("Reset failed!")
    rt1_response = requests.post("http://localhost:8000/predict", json=data)
    while True:
        if rt1_response.status_code == 200:
            action = rt1_response.json()
            """
            {'world_vector': [-0.00391387939453125, -0.00391387939453125, -0.00391387939453125], 
            'rotation_delta': [-0.0030739307403564453, -0.0030739307403564453, -0.0030739307403564453], 
            'terminate_episode': [1, 0, 0], 
            'gripper_closedness_action': [-0.001956939697265625]}
            """
            world_vector = action["world_vector"]
            rotation_delta = action["rotation_delta"]
            terminate_episode = action["terminate_episode"]
            gripper_closedness_action = action["gripper_closedness_action"]
            action = np.concatenate(
                [world_vector, rotation_delta, gripper_closedness_action]
            )
            print(f"Action: {np.round(action, 3)}")
            action[:3] = action[:3] * 100
            action[3:6] = action[3:6] * 10
            # print(f"Action: {np.round(action, 3)}")
            obs, reword, terminat = env.apply_action(action=action, relative_mode=True)
            front_rgb = obs["camera"]["wrist_rgb"]
            data = prepare_date(front_rgb)
            rt1_response = requests.post("http://localhost:8000/predict", json=data)
        else:
            print("Request failed!")
            break
