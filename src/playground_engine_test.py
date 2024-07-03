from pyvirtualdisplay import Display
import torch, os
import open3d as o3d

from yaml_config_utils import get_config, load_config
from interfaces import setup_LMP, LMP_interface
from visualizers import ValueMapVisualizer
from utils import set_lmp_objects
import numpy as np
import engine_interfaces
from envs.pyrep_env.pyrep_quad_env import VoxPoserPyRepQuadcopterEnv
from envs.ros_env.ros_env import VoxPoserROSDroneEnv
from envs.dummy_env import DummyEnv
from engine_interfaces import Dummy
from VLMPipline.VLM import VLM

torch.set_grad_enabled(False)
disp = Display(visible=False, size=(1920, 1080))
disp.start()
# vlm config
owlv2_model_path = os.path.expanduser("~/models/google-owlv2-large-patch14-finetuned")
owlv2_model_path = os.path.expanduser("~/models/google-owlv2-base-patch16-ensemble")
sam_model_path = os.path.expanduser("~/models/facebook-sam-vit-huge")
# sam_model_path = os.path.expanduser("~/models/facebook-sam-vit-base")
xmem_model_path = os.path.expanduser("~/models/XMem.pth")
resnet_18_path = os.path.expanduser("~/models/resnet18.pth")
resnet_50_path = os.path.expanduser("~/models/resnet50.pth")
config_path = "./src/configs/pyrep_quadcopter.yaml"
# config_path = "./src/configs/airsim_ros_quadcopter.yaml"
scene_path = "./src/scene/quadcopter_tree_sofa_helicopter.ttt"
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
prefix = "/shared/codes/VoxPoser"

# sparkv3_engine_config = load_config(os.path.join(prefix,"src/configs/sparkv3_config.yaml"))
# sparkv35_engine_config = load_config(os.path.join(prefix,"src/configs/sparkv3_5_config.yaml"))
# erniev4_engine_config = load_config(os.path.join(prefix,"src/configs/ERNIEv4_config.yaml"))
tgi_config = load_config(
    os.path.join(prefix, "src/configs/TGI_deepseek-coder-6.7B-instruct-AWQ.yaml")
)
tgi_config33 = load_config(
    os.path.join(prefix, "src/configs/TGI_deepseek-coder-33B-instruct-AWQ.yaml")
)
# engine_erniev4 = getattr(engine_interfaces, erniev4_engine_config["type"])(
#     **erniev4_engine_config
# )  # engine initialization
# engine_sparkv3 = getattr(engine_interfaces, sparkv3_engine_config["type"])(
#     **sparkv35_engine_config
# )  # engine initialization
engine_tgi_deepseek = getattr(engine_interfaces, tgi_config["type"])(
    **tgi_config
)  # engine initialization
engine_tgi_deepseek33 = getattr(engine_interfaces, tgi_config["type"])(
    **tgi_config33
)  # engine initialization

visualizer = ValueMapVisualizer(env_config["visualizer"])
# env = VoxPoserPyRepQuadcopterEnv(
#     visualizer=visualizer,
#     headless=True,
#     coppelia_scene_path=scene_path,
#     vlmpipeline=vlmpipeline,
#     target_objects=scene_target_objects,
# )
# env = VoxPoserROSDroneEnv(vlmpipeline=vlmpipeline, visualizer=visualizer)
env = DummyEnv(vlmpipeline=vlmpipeline, visualizer=visualizer)
descriptions, obs = env.reset()
env.target_objects.append("apple")
# descriptions = "Fly to the tank with a soldier standing next to it, keeping a distance of at least 3 meters"
descriptions = "fly to the apple, then fly to the point where you started"
lmps, lmp_env = setup_LMP(
    env, env_config, debug=True, engine_call_fn=engine_tgi_deepseek
)
voxposer_ui = lmps["plan_ui"]
set_lmp_objects(lmps, env.get_object_names())
# try:
voxposer_ui(descriptions)
# except Exception as e:
# print(f"{type(e)}: {e}")
if type(env) == VoxPoserPyRepQuadcopterEnv:
    env._pyrep.stop()
    env._pyrep.shutdown()
disp.stop()
import os,re,time
import engine_interfaces
from yaml_config_utils import get_config, load_config

def split_prompt(prompt):
    if False:
        pattern = r"(objects = .*?\n# Query:.*?(?:$|\n))"
    else:
        pattern = r"(\n# Query:.*?(?:$|\n))"
    matches = re.split(pattern, prompt, flags=re.DOTALL)
    return [matches[0] + "\n\n" + matches[1]] + matches[2:-1]

prefix ="/shared/codes/VoxPoser"
raw_prompt = ''
with open(os.path.join(prefix,'src/prompts/pyrep_quadcopter/real_get_affordance_map_prompt.txt')) as f:
    raw_prompt = f.read()
descriptions = "fly to the table, then fly to the tree, and at last fly to the sofa"
descriptions = "# Query: a point 10cm above the sky."
user_prompt = raw_prompt + f"\n{descriptions}"
splited_prompt = split_prompt(raw_prompt)
    
print(splited_prompt)
tgi_config33 = load_config(
    os.path.join(prefix,"src/configs/TGI_deepseek-coder-33B-instruct-AWQ.yaml")
)
ernieht_engine_config = load_config(os.path.join(prefix,"src/configs/ERNIEht.yaml"))
engine_erniev4 = getattr(engine_interfaces, ernieht_engine_config["type"])(
    **ernieht_engine_config
)  # engine initialization

_stop_tokens = ['# Query: ','objects = ','# done']
temperature = 0.3
use_cache = False
max_tokens = 512

engine_tgi_deepseek33 = getattr(engine_interfaces, tgi_config33["type"])(
    **tgi_config33
)  # engine initialization

start_time = time.time()
ret = engine_erniev4(prompt = (user_prompt,splited_prompt), stop = _stop_tokens, temperature = temperature, max_tokens = max_tokens, use_cache = use_cache)
end_time = time.time()
print(ret)
print(f"time cost:{end_time-start_time}")