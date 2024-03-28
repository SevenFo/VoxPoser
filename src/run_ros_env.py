import rospy, torch
from envs.ros_env.ros_env import VoxPoserROSDroneEnv
from VLMPipline.VLM import VLM
if __name__ == '__main__':
    # vlm config
    with torch.no_grad():
        owlv2_model_path = "/models/google-owlv2-large-patch14-finetuned"
        owlv2_model_path = "/models/google-owlv2-base-patch16-ensemble"
        sam_model_path = "/models/facebook-sam-vit-huge"
        # sam_model_path = "/models/facebook-sam-vit-base"
        xmem_model_path = "/models/XMem.pth"
        resnet_18_path = "/models/resnet18.pth"
        resnet_50_path = "/models/resnet50.pth"
        
        rospy.init_node('voxposer_ros_drone_env')
        vlmpipeline = VLM(
            owlv2_model_path,
            sam_model_path,
            xmem_model_path,
            resnet_18_path,
            resnet_50_path,
            verbose=True,
            resize_to=[640, 640],
            verbose_frame_every=1,
            input_batch_size=1,
        )
        env = VoxPoserROSDroneEnv(vlmpipeline=vlmpipeline)
        env.reset()
        env.get_3d_obs_by_name_by_vlm("Stone lion statue")
        # rospy.spin()
        # env.takeoff()
        # env.land()
        # env.move(0.1, 0.1, 0.1, 0.1)
        # env.get_pose()