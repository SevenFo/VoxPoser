from pyvirtualdisplay import Display
import torch
import os
import time
import open3d as o3d

from yaml_config_utils import get_config, load_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from utils import set_lmp_objects
import numpy as np
import engine_interfaces
from envs.ros_env.ros2_drone_env import VoxPoserROS2DroneEnv
from engine_interfaces import Dummy
from VLMPipline.VLM import VLM
from VLMPipline.VLMM import VLMProcessWrapper
import rclpy
import threading
torch.set_grad_enabled(False)

if __name__ == "__main__":
    # disp = Display(visible=False, size=(1920, 1080))
    # disp.start()
    # vlm config
    owlv2_model_path = "/root/models/google-owlv2-large-patch14-finetuned"
    owlv2_model_path = "/root/models/google-owlv2-base-patch16-ensemble"
    sam_model_path = "/root/models/facebook-sam-vit-huge"
    xmem_model_path = "/root/models/XMem.pth"
    resnet_18_path = "/root/models/resnet18.pth"
    resnet_50_path = "/root/models/resnet50.pth"
    config_path = "./src/configs/pyrep_quadcopter.yaml"
    config_path = "./src/configs/airsim_ros_quadcopter.yaml"
    config_path = "./src/configs/isaac_ros2_quadcopter_v2.yaml"
    scene_target_objects = [
        "pumpkin",
        "house",
        "apple",
        "Stone lion statue",
        "windmill",
    ]

    env_config = get_config(config_path=config_path)

    log_dir = os.path.join(
        env_config.log_dir,
        f"{time.strftime('%Y-%m-%d-%H-%M-%S')}",
    )
    env_config["visualizer"]["save_dir"] = os.path.join(log_dir, "visualizer")
    input_shape = env_config.vlm["input_shape"]
    batch_size = env_config.vlm["batch_size"]

    if "scene_target_objects" in env_config.env:
        scene_target_objects = env_config.env["scene_target_objects"]

    vlmpipeline = VLMProcessWrapper(
        scene_target_objects,
        (batch_size,) + tuple(input_shape),
        owlv2_model_path,
        sam_model_path,
        xmem_model_path,
        resnet_18_path,
        resnet_50_path,
        resize_to=[640, 640],
        category_multiplier=100,
        verbose=True,
        verbose_frame_every=1,
        verbose_to_disk=True,
        log_dir=log_dir,
        input_batch_size=batch_size,
    )
    vlmpipeline.start()
    prefix = "/shared/codes/VoxPoser.worktrees/VoxPoser"

    ollama_config = load_config(os.path.join(prefix, "src/configs/TGI_deepseek-coder-33B-instruct-AWQ.yaml"))
    ollama_config = load_config(os.path.join(prefix, "src/configs/TGI_deepseek-coder-33B-instruct-AWQ.yaml"))
    ollama_config = load_config(os.path.join(prefix, "src/configs/ollama_config.yaml"))
    ollama_config = load_config(os.path.join(prefix, "src/configs/openai_config.yaml"))

    engine_ollama_deepseek33_q4 = getattr(engine_interfaces, ollama_config["type"])(
        **ollama_config
    )

    visualizer = ValueMapVisualizer(env_config["visualizer"])
    
    # rclpy.init(args=['--ros-args', '--param', 'use_sim_time:=true'])
    rclpy.init()
    env = VoxPoserROS2DroneEnv(
        vlmpipeline=vlmpipeline,
        visualizer=visualizer,
        target_objects=scene_target_objects,
        configs=env_config.env,
    )
    env_thread = threading.Thread(target=rclpy.spin, args=(env, ), daemon=True)
    env_thread.start()

    descriptions, obs = env.reset()


    if "description" in env_config.env:
        descriptions = env_config.env["description"]

    lmps, lmp_env = setup_LMP(
        env, env_config, debug=False, engine_call_fn=engine_ollama_deepseek33_q4
    )
    voxposer_ui = lmps["plan_ui"]
    set_lmp_objects(lmps, env.get_object_names())
    # try:
    voxposer_ui(descriptions)
    # except Exception as e:
    #     print(f"{type(e)}: {e}")
    # env._pyrep.stop()
    # env._pyrep.shutdown()
    # disp.stop()
    env.destroy_node()
    rclpy.shutdown()