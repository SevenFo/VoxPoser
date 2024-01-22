# -*- coding: utf-8 -*-

import openai
from pyvirtualdisplay import Display
import torch

from arguments import get_config, load_config
from interfaces import setup_LMP, LMP_interface
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects
import numpy as np
from rlbench import tasks
import engine_interfaces

from VLMPipline.VLM import VLM
torch.set_grad_enabled(False)
disp = Display(visible=False, size=(1920, 1080))
disp.start()
openai.api_key = None  # set your API key here

# load other config file (LMP, visualization, visual env, etc.)
config = get_config("rlbench")
# load engine config file (spark v3)
sparkv3_engine_config = load_config(
    "/mnt/workspace/src/configs/sparkv3_config.yaml"
)
erniev4_engine_config = load_config(
    "/mnt/workspace/src/configs/ERNIEv4_config.yaml"
)
print(erniev4_engine_config)

# vlm config
owlv2_model_path = "/models/google-owlv2-large-patch14-finetuned"
owlv2_model_path = "/models/google-owlv2-base-patch16-ensemble"
sam_model_path = "/models/facebook-sam-vit-huge"
# sam_model_path = "/models/facebook-sam-vit-base"
xmem_model_path = "/models/XMem.pth"
resnet_18_path = "/models/resnet18.pth"
resnet_50_path = "/models/resnet50.pth"

vlmpipeline = VLM(
    owlv2_model_path,
    sam_model_path,
    xmem_model_path,
    resnet_18_path,
    resnet_50_path,
    verbose=False,
    resize_to=[640,640]
)

# uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
# for lmp_name, cfg in config['lmp_config']['lmps'].items():
#     cfg['model'] = 'gpt-3.5-turbo'

# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config["visualizer"])
env = VoxPoserRLBench(visualizer=visualizer, headless=True, vlmpipeline=vlmpipeline, scene_obs_size=640)
engine_sparkv3 = getattr(engine_interfaces, sparkv3_engine_config["type"])(
    **sparkv3_engine_config
)  # engine initialization
engine_erniev4 = getattr(engine_interfaces, erniev4_engine_config["type"])(
    **erniev4_engine_config
)  # engine initialization
lmps, lmp_env = setup_LMP(env, config, debug=False, engine_call_fn=engine_erniev4)
voxposer_ui = lmps["plan_ui"]

# below are the tasks that have object names added to the "task_object_names.json" file
# uncomment one to use
env.load_task(tasks.PutRubbishInBin)
# env.load_task(tasks.LampOff)
# env.load_task(tasks.OpenWineBottle)
# env.load_task(tasks.PushButton)
# env.load_task(tasks.TakeOffWeighingScales)
# env.load_task(tasks.MeatOffGrill)
# env.load_task(tasks.SlideBlockToTarget)
# env.load_task(tasks.TakeLidOffSaucepan)
# env.load_task(tasks.TakeUmbrellaOutOfUmbrellaStand)
descriptions, obs = env.reset()
controller_config = config["controller"]
planner_config = config["planner"]
lmp_env_config = config["lmp_config"]["env"]
lmps_config = config["lmp_config"]["lmps"]
env_name = config["env_name"]
# LMP env wrapper
lmp_env = LMP_interface(
    env, lmp_env_config, controller_config, planner_config, env_name=env_name
)
# object = "tomato"
# result = lmp_env.detect(object)
# pcd_vector = o3d.geometry.PointCloud()
# pcd_vector.points = o3d.utility.Vector3dVector(result['_point_cloud_world'])
# origin = pcd_vector.get_center()
# coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2,origin=origin)
# o3d.visualization.draw_geometries([pcd_vector,coordinate])
# o3d.io.write_point_cloud(f'{object}_pcd.pcd',pcd_vector)
# print(result)

set_lmp_objects(
    lmps, env.get_object_names()
)  # set the object names to be used by voxposer

# 关于task description从哪里来的: rlbench 内置了一些task，并且这些task有多种描述，ref:https://github.com/stepjam/RLBench/blob/master/tutorials/simple_task.md
instruction = np.random.choice(descriptions)
# instruction = "throw away the trash, leaving any other objects alone"
voxposer_ui(instruction)
disp.stop()
