from pyvirtualdisplay import Display
from pyvirtualdisplay import Display
import torch
import open3d as o3d

from arguments import get_config, load_config
from interfaces import setup_LMP, LMP_interface
from visualizers import ValueMapVisualizer
from utils import set_lmp_objects
import numpy as np
import engine_interfaces
from envs.pyrep_quad_env import VoxPoserPyRepQuadcopterEnv
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
scene_path = "./scene/quadcopter.ttt"
config = get_config(config_path=config_path)
vlmpipeline = VLM(
    owlv2_model_path,
    sam_model_path,
    xmem_model_path,
    resnet_18_path,
    resnet_50_path,
    verbose=False,
    resize_to=[320,320],
    input_batch_size=5
)

visualizer = ValueMapVisualizer(config['visualizer'])
env = VoxPoserPyRepQuadcopterEnv(visualizer=visualizer,headless=True,coppelia_scene_path=scene_path,vlmpipeline=vlmpipeline)
descriptions, obs = env.reset()
dummy_engine = Dummy()
lmps, lmp_env = setup_LMP(env, config, debug=False, engine_call_fn=dummy_engine.__call__)
voxposer_ui = lmps['plan_ui']
set_lmp_objects(lmps, env.get_object_names())

voxposer_ui(env.descriptions[-1])

env._pyrep.stop()
env._pyrep.shutdown()

disp.stop()
