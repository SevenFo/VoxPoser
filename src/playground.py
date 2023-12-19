# -*- coding: utf-8 -*-

import openai
from arguments import get_config, load_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects
import numpy as np
from rlbench import tasks
import engine_interfaces

openai.api_key = None  # set your API key here

# load other config file (LMP, visualization, visual env, etc.)
config = get_config("rlbench")
# load engine config file (spark v3)
sparkv3_engine_config = load_config(
    "/root/workspace/VoxPoser/src/configs/sparkv3_config.yaml"
)
erniev4_engine_config = load_config(
    "/root/workspace/VoxPoser/src/configs/ERNIEv4_config.yaml"
)
print(erniev4_engine_config)

# uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
# for lmp_name, cfg in config['lmp_config']['lmps'].items():
#     cfg['model'] = 'gpt-3.5-turbo'

# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config["visualizer"])
env = VoxPoserRLBench(visualizer=visualizer)
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
set_lmp_objects(
    lmps, env.get_object_names()
)  # set the object names to be used by voxposer

# 关于task description从哪里来的: rlbench 内置了一些task，并且这些task有多种描述，ref:https://github.com/stepjam/RLBench/blob/master/tutorials/simple_task.md
instruction = np.random.choice(descriptions)
# instruction = "throw away the trash, leaving any other objects alone"
voxposer_ui(instruction)
