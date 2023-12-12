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
engine_config = load_config("/root/workspace/VoxPoser/src/configs/sparkv3_config.yaml")

# uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
# for lmp_name, cfg in config['lmp_config']['lmps'].items():
#     cfg['model'] = 'gpt-3.5-turbo'

# initialize env and voxposer ui
visualizer = ValueMapVisualizer(config["visualizer"])
env = VoxPoserRLBench(visualizer=visualizer)
engine = getattr(engine_interfaces, engine_config.pop("name"))(
    **engine_config
)  # engine initialization
lmps, lmp_env = setup_LMP(env, config, debug=False, engine_call_fn=engine)
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

instruction = np.random.choice(descriptions)
voxposer_ui(instruction)
