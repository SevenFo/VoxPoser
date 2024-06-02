from pyvirtualdisplay import Display
import torch, os

from yaml_config_utils import get_config, load_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from utils import set_lmp_objects
import engine_interfaces
from envs.ros_env.ros_env import VoxPoserROSDroneEnv
from VLMPipline.VLMM import VLMProcessWrapper

torch.set_grad_enabled(False)
os.environ["ROS_MASTER_URI"] = "http://localhost:11311"
os.environ["ROS_IP"] = "localhost"

if __name__ == "__main__":
    # disp = Display(visible=False, size=(1920, 1080))
    # disp.start()
    # vlm config
    owlv2_model_path = "/models/google-owlv2-large-patch14-finetuned"
    owlv2_model_path = "/models/google-owlv2-base-patch16-ensemble"
    sam_model_path = "/models/facebook-sam-vit-huge"
    # sam_model_path = "/models/facebook-sam-vit-base"
    xmem_model_path = "/models/XMem.pth"
    resnet_18_path = "/models/resnet18.pth"
    resnet_50_path = "/models/resnet50.pth"
    config_path = "./src/configs/pyrep_quadcopter.yaml"
    config_path = "./src/configs/airsim_ros_quadcopter.yaml"
    scene_target_objects = [
        "pumpkin",
        "house",
        "apple",
        "Stone lion statue",
        "windmill",
    ]
    # scene_target_objects = [
    #     "fire extinguisher",
    # ]
    env_config = get_config(config_path=config_path)
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
        log_dir=env_config.log_dir,
        input_batch_size=batch_size,
    )
    vlmpipeline.start()
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

    env = VoxPoserROSDroneEnv(
        vlmpipeline=vlmpipeline,
        visualizer=visualizer,
        target_objects=scene_target_objects,
        configs=env_config.env,
    )

    descriptions, obs = env.reset()
    # descriptions = "fly to the table, then fly to the tree, and at last fly to the sofa"
    descriptions = (
        "fly to the house, then fly to the point where you started"  # checked
    )
    # descriptions = "fly to the fire extinguisher"  # "  # underchecking
    # descriptions = "fly forward 100cm"
    # descriptions = "fly forward 100cm, then fly to the Stone lion statue, and at last fly backward 300cm" # underchecking
    if "description" in env_config.env:
        descriptions = env_config.env["description"]

    lmps, lmp_env = setup_LMP(
        env, env_config, debug=False, engine_call_fn=engine_tgi_deepseek33
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
