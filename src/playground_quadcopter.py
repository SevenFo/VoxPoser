from pyvirtualdisplay import Display
from pyvirtualdisplay import Display
import torch
import open3d as o3d

from yaml_config_utils import get_config, load_config
from interfaces import setup_LMP, LMP_interface
from visualizers import ValueMapVisualizer
from utils import set_lmp_objects
import numpy as np
import engine_interfaces
from envs.pyrep_env.pyrep_quad_env import VoxPoserPyRepQuadcopterEnv
from engine_interfaces import Dummy
from VLMPipline.VLM import VLM

torch.set_grad_enabled(False)
disp = Display(visible=False, size=(1920, 1080))
disp.start()
# vlm config
owlv2_model_path = "/models/google-owlv2-large-patch14-finetuned"
owlv2_model_path = "/models/google-owlv2-base-patch16-ensemble"
sam_model_path = "/models/facebook-sam-vit-huge"
# sam_model_path = "/models/facebook-sam-vit-base"
xmem_model_path = "/models/XMem.pth"
resnet_18_path = "/models/resnet18.pth"
resnet_50_path = "/models/resnet50.pth"
config_path = "configs/pyrep_quadcopter.yaml"
scene_path = "./scene/quadcopter_tree_sofa_helicopter.ttt"
scene_target_objects = ["quadcopter", "sofa", "tree", "helicopter", "table"]
env_config = get_config(config_path=config_path)
vlmpipeline = VLM(
    owlv2_model_path,
    sam_model_path,
    xmem_model_path,
    resnet_18_path,
    resnet_50_path,
    verbose=False,
    resize_to=[480, 480],
    verbose_frame_every=1,
    input_batch_size=5,
)
sparkv3_engine_config = load_config("/mnt/workspace/src/configs/sparkv3_config.yaml")
sparkv35_engine_config = load_config("/mnt/workspace/src/configs/sparkv3_5_config.yaml")
erniev4_engine_config = load_config("/mnt/workspace/src/configs/ERNIEv4_config.yaml")
tgi_config = load_config(
    "/mnt/workspace/src/configs/TGI_deepseek-coder-6.7B-instruct-AWQ.yaml"
)
engine_erniev4 = getattr(engine_interfaces, erniev4_engine_config["type"])(
    **erniev4_engine_config
)  # engine initialization
engine_sparkv3 = getattr(engine_interfaces, sparkv3_engine_config["type"])(
    **sparkv35_engine_config
)  # engine initialization
engine_tgi_deepseek = getattr(engine_interfaces, tgi_config["type"])(
    **tgi_config
)  # engine initialization

visualizer = ValueMapVisualizer(env_config["visualizer"])
env = VoxPoserPyRepQuadcopterEnv(
    visualizer=visualizer,
    headless=True,
    coppelia_scene_path=scene_path,
    vlmpipeline=vlmpipeline,
    target_objects=scene_target_objects,
)
descriptions, obs = env.reset()
descriptions = "fly to the table, then fly to the tree, and at last fly to the sofa"
lmps, lmp_env = setup_LMP(
    env, env_config, debug=False, engine_call_fn=engine_tgi_deepseek
)
voxposer_ui = lmps["plan_ui"]
set_lmp_objects(lmps, env.get_object_names())

voxposer_ui(descriptions)
env._pyrep.stop()
env._pyrep.shutdown()
disp.stop()
