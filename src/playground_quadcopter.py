from pyvirtualdisplay import Display
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
# from envs.ros_env.ros_env import VoxPoserROSDroneEnv
from engine_interfaces import Dummy
from VLMPipline.VLM import VLM
from VLMPipline.VLMM import VLMProcessWrapper

os.environ["HOME"] = "/"
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    disp = Display(visible=False, size=(1920, 1080))
    disp.start()
    # vlm config
    owlv2_model_path = os.path.expanduser(
        "~/models/google-owlv2-large-patch14-finetuned"
    )
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
    vlmpipeline = VLMProcessWrapper(
        scene_target_objects,
        (5, 3, 480, 480),
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
    vlmpipeline.start()
    # vlmpipeline = VLM(
    #     owlv2_model_path,
    #     sam_model_path,
    #     xmem_model_path,
    #     resnet_18_path,
    #     resnet_50_path,
    #     verbose=False,
    #     resize_to=[480, 480],
    #     verbose_frame_every=1,
    #     input_batch_size=5,
    # )
    # vlmpipeline = VLM(
    #     owlv2_model_path,
    #     sam_model_path,
    #     xmem_model_path,
    #     resnet_18_path,
    #     resnet_50_path,
    #     verbose=False,
    #     resize_to=[640, 640],
    #     verbose_frame_every=1,
    #     input_batch_size=5,
    # )
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
    env = VoxPoserPyRepQuadcopterEnv(
        visualizer=visualizer,
        headless=True,
        coppelia_scene_path=scene_path,
        vlmpipeline=vlmpipeline,
        target_objects=scene_target_objects,
    )
    # env = VoxPoserROSDroneEnv(vlmpipeline=vlmpipeline, visualizer=visualizer)
    descriptions, obs = env.reset()
    descriptions = "fly to the table, then fly to the tree, and at last fly to the sofa"
    # descriptions = "fly to the apple, then fly to the point where you started"
    lmps, lmp_env = setup_LMP(
        env, env_config, debug=False, engine_call_fn=engine_tgi_deepseek33
    )
    voxposer_ui = lmps["plan_ui"]
    set_lmp_objects(lmps, env.get_object_names())
    try:
        voxposer_ui(descriptions)
        env._pyrep.stop()
        env._pyrep.shutdown()
        env.vlm.shutdown()
        disp.stop()

    except Exception as e:
        print(f"{type(e)}: {e}")
        env._pyrep.stop()
        env._pyrep.shutdown()
        env.vlm.shutdown()
        disp.stop()
