from pyvirtualdisplay import Display
import torch
import os
import time

from yaml_config_utils import get_config, load_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from utils import set_lmp_objects
import engine_interfaces
from envs.ros_env.ros_env import VoxPoserROSDroneEnv
from VLMPipline.VLMM import VLMProcessWrapper

torch.set_grad_enabled(False)

# ROS 节点配置
os.environ["ROS_MASTER_URI"] = "http://192.168.1.105:11311" # master
os.environ["ROS_IP"] = "192.168.1.213" # slave (本机)

if __name__ == "__main__":

    # vlm config
    owlv2_model_path = "/home/randuser/models/google-owlv2-large-patch14-finetuned"
    owlv2_model_path = "/home/randuser/models/google-owlv2-base-patch16-ensemble"
    sam_model_path = "/home/randuser/models/facebook-sam-vit-huge"
    xmem_model_path = "/home/randuser/models/XMem.pth"
    resnet_18_path = "/home/randuser/models/resnet18.pth"
    resnet_50_path = "/home/randuser/models/resnet50.pth"
    config_path = "./src/configs/real_ros_quadcopter.yaml"
    scene_target_objects = []

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
    prefix = "/shared/codes/VoxPoser"

    tgi_config = load_config(os.path.join(prefix, "src/configs/TGI_deepseek-coder-33B-instruct-AWQ.yaml"))

    tgi = getattr(engine_interfaces, tgi_config["type"])(
        **tgi_config
    )
    
    visualizer = ValueMapVisualizer(env_config["visualizer"])

    env = VoxPoserROSDroneEnv(
        vlmpipeline=vlmpipeline,
        visualizer=visualizer,
        target_objects=scene_target_objects,
        configs=env_config.env,
    )

    descriptions, obs = env.reset()

    descriptions = (
        "fly to the house, then fly to the point where you started"  # checked
    )

    if "description" in env_config.env:
        descriptions = env_config.env["description"]

    lmps, lmp_env = setup_LMP(
        env, env_config, debug=False, engine_call_fn=tgi
    )
    voxposer_ui = lmps["plan_ui"]
    set_lmp_objects(lmps, env.get_object_names())

    voxposer_ui(descriptions)

