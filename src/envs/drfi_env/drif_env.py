from utils import normalize_vector, bcolors, Observation
from visualizers import ValueMapVisualizer
from VLMPipline.VLM import VLM
from VLMPipline.utils import convert_depth_to_pointcloud
from envs.pyrep_env.pyrepqudacopter import PyRepQuadcopter


class VoxPoserDRIFEnv:
    def __init__(
        self,
        visualizer: ValueMapVisualizer = None,
        headless=False,
        vlmpipeline: VLM = None,
        target_objects=["quadcopter", "table"],
    ):
        # start the simulation and initialize the environment
        self._target_objects = target_objects
        ZeroDivisionError
