import rospy, threading, os
import sensor_msgs.point_cloud2 as pl2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
import tf2_ros
import tf
import tf2_geometry_msgs, tf2_sensor_msgs
import cv_bridge
import numpy as np
from VLMPipline.VLM import VLM
import open3d as o3d
import warnings
import cv2
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3, Twist
import actionlib
from actionlib_msgs.msg import GoalStatus
import transforms3d
from std_srvs.srv import SetBool, SetBoolResponse
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool, Process

from utils import normalize_vector, bcolors, Observation
 
WIDTH = 640
HEIGHT = 640

class VoxPoserROSDroneEnv:
    def __init__(self, vlmpipeline: VLM = None, visualizer=None, use_old_airsim = True):
        print(f"current process ID: {os.getpid()}")
        self._use_old_airsim = use_old_airsim
        self.latest_obs = {}
        self.lookat_vectors = {}
        self.camera_params = {}
        self._cvb = cv_bridge.CvBridge()
        self.vlm = vlmpipeline
        self.camera_names = ["front_center","front_left","front_right","down_center","rear_center"]
        self.visualizer = visualizer
        self.workspace_bounds_min = np.array([0, 0, 0])
        self.workspace_bounds_max = np.array([5, 5, 3])
        if self.visualizer is not None:
            self.visualizer.update_bounds(
                self.workspace_bounds_min, self.workspace_bounds_max
            )
            
        self.init_ros()
        self.init_task()

        
    def init_task(self):
        self.descriptions = [
            "fly around a distance above the table",
            "From under the table, cross the past to the 100cm in front of the table, then fly to the top 100cm above the table",
            "fly to the table",
            "go to the table",
        ]
        self.target_objects = ['pumpkin', "house", "apple", "Stone lion statue", "windmill"]
        print(f"we only process these objects: {self.target_objects}")
        self.category_multiplier = 100  # which means the instance number of each object categery is less than 100
        self.name2categerylabel = {
        name: i
        for i, name in enumerate(
            self.target_objects, start=1
        )  # the category label start from 1 as 0 represent background
        }
        self.categerylabel2name = {
            i: name
            for i, name in enumerate(
                self.target_objects, start=1
            )  # the category label start from 1 as 0 represent background
        }
        rospy.loginfo("wait for /airsim_node/reset service")
        rospy.wait_for_service('/airsim_node/reset')
        rospy.loginfo(f"reset env: {rospy.ServiceProxy('/airsim_node/reset', SetBool)(SetBool._request_class(True)).message}")
        
    def init_ros(self):
        # 初始化参数
        rospy.set_param('/airsim_node/world_frame_id', 'world_ned')
        rospy.set_param('/airsim_node/ros_frame_id', 'world_enu')
        rospy.set_param('/airsim_node/odom_frame_id', 'odom_local_ned')
        rospy.set_param('/airsim_node/drone_base_link_frame_id', 'base_link')
        rospy.set_param('/airsim_node/coordinate_system_enu', False)
        rospy.set_param('/airsim_node/update_airsim_control_every_n_sec', 0.01)
        rospy.set_param('/airsim_node/update_airsim_img_response_every_n_sec', 0.08)
        rospy.set_param('/airsim_node/publish_clock', False)
        rospy.set_param('/airsim_node/action_max_wait_time', 30)
        self._world_frame_id = rospy.get_param('/airsim_node/world_frame_id')
        self._ros_frame_id = rospy.get_param('/airsim_node/ros_frame_id')
        self._odom_frame_id = rospy.get_param('/airsim_node/odom_frame_id')
        self._drone_base_link_frame_id = rospy.get_param('/airsim_node/drone_base_link_frame_id')
        airsim_client_host = rospy.get_param('/airsim_node/airsim_client_host', 'localhost')
        airsim_client_port = rospy.get_param('/airsim_node/airsim_client_port', 10000)
        update_airsim_img_response_every_n_sec = rospy.get_param('/airsim_node/update_airsim_img_response_every_n_sec', 1)

        self._cmd_pub = rospy.Publisher('/airsim_node/vel_cmd_world_frame', Twist, queue_size=1)
        self._point_front_center_sub = rospy.Subscriber('/airsim_node/drone_1/front_center/1/registered/points_transformed', PointCloud2,self.point_front_center_sub_callback)
        self._odom_sub = rospy.Subscriber('/airsim_node/drone_1/odom_local_enu', Odometry, self.odom_sub_callback)

        self.rgb_image_front_center_sub = rospy.Subscriber('/airsim_node/drone_1/front_center/0/image_raw', Image, self.rgb_image_sub_front_center_callback)
        self.cloudpoint_front_center_sub = rospy.Subscriber('/airsim_node/drone_1/front_center/1/registered/points_transformed',PointCloud2, self.cloudpoint_sub_front_center_callback)
        self.camera_info_front_center_sub = rospy.Subscriber('/airsim_node/drone_1/front_center/1/camera_info', CameraInfo, self.camera_info_sub_front_center_callback)
        self.rgb_image_front_left_sub = rospy.Subscriber('/airsim_node/drone_1/front_left/0/image_raw', Image, self.rgb_image_sub_front_left_callback)
        self.cloudpoint_front_left_sub = rospy.Subscriber('/airsim_node/drone_1/front_left/1/registered/points_transformed',PointCloud2, self.cloudpoint_sub_front_left_callback)
        self.camera_info_front_left_sub = rospy.Subscriber('/airsim_node/drone_1/front_left/1/camera_info', CameraInfo, self.camera_info_sub_front_left_callback)
        self.rgb_image_front_right_sub = rospy.Subscriber('/airsim_node/drone_1/front_right/0/image_raw', Image, self.rgb_image_sub_front_right_callback)
        self.cloudpoint_front_right_sub = rospy.Subscriber('/airsim_node/drone_1/front_right/1/registered/points_transformed',PointCloud2, self.cloudpoint_sub_front_right_callback)
        self.camera_info_front_right_sub = rospy.Subscriber('/airsim_node/drone_1/front_right/1/camera_info', CameraInfo, self.camera_info_sub_front_right_callback)
        self.rgb_image_down_center_sub = rospy.Subscriber('/airsim_node/drone_1/down_center/0/image_raw', Image, self.rgb_image_sub_down_center_callback)
        self.cloudpoint_down_center_sub = rospy.Subscriber('/airsim_node/drone_1/down_center/1/registered/points_transformed',PointCloud2, self.cloudpoint_sub_down_center_callback)
        self.camera_info_down_center_sub = rospy.Subscriber('/airsim_node/drone_1/down_center/1/camera_info', CameraInfo, self.camera_info_sub_down_center_callback)
        self.rgb_image_rear_center_sub = rospy.Subscriber('/airsim_node/drone_1/rear_center/0/image_raw', Image, self.rgb_image_sub_rear_center_callback)
        self.cloudpoint_rear_center_sub = rospy.Subscriber('/airsim_node/drone_1/rear_center/1/registered/points_transformed',PointCloud2, self.cloudpoint_sub_rear_center_callback)
        self.camera_info_rear_center_sub = rospy.Subscriber('/airsim_node/drone_1/rear_center/1/camera_info', CameraInfo, self.camera_info_sub_rear_center_callback)
        rospy.init_node("voxposer")
        
        self._action_client = actionlib.SimpleActionClient('/airsim_node/goto', MoveBaseAction)

        
    def point_front_center_sub_callback(self, msg):
        point = np.array(list(pl2.read_points(msg, field_names = ('x', 'y', 'z'), skip_nans=True)))
        # print(point.shape)
        self.latest_obs.update({
            'front_center_cloudpoint': point
        })   
        
    def get_object_names(self):
        return self.target_objects
    
    def odom_sub_callback(self, msg):
        pose = msg.pose.pose
        twist = msg.twist.twist
        self.latest_obs.update({
            'quad_pose': np.array([pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ]+list(transforms3d.euler.quat2euler([pose.orientation.w,pose.orientation.x, pose.orientation.y, pose.orientation.z])))
        })

    def _get_rgb_frames(self):
        rgb_frames = {}  # in c w h
        for cam in self.camera_names:
            while f"{cam}_rgb" not in self.latest_obs:
                # raise ValueError(f"no {cam}_rgb found")
                rospy.loginfo("waiting for rgb image")
                rospy.sleep(0.1)
            rgb_frames[cam] = self.latest_obs[f"{cam}_rgb"].transpose(
                [2, 0, 1]
            )
        frames = np.stack(list(rgb_frames.values()), axis=0)
        return frames
    
    def reset(self):
        descriptions = self.descriptions[0]
        self.init_obs = self.latest_obs
        self.latest_mask = {}
        self.vlm.reset()
        self.target_objects_labels = list(range(len(self.target_objects)))
        frames = self._get_rgb_frames()
        masks = self.vlm.process_first_frame(
            self.target_objects, frames, verbose=True, owlv2_threshold=0.2
        )
        if not np.any(masks):
            raise ValueError(
                "no intrested object found in the scene, may be you should let robot turn around or change the scene or change the target object"
            )
            return None
        [
            self.latest_mask.update({cam: mask})
            for cam, mask in zip(self.camera_names, masks)
        ]
        self._update_visualizer()
        self.has_processed_first_frame = True
        return descriptions, self.latest_obs
    
    def get_3d_obs_by_name_by_vlm(self, query_name, cameras=None):
            """
            Retrieves 3D point cloud observations and normals of an object by its name by VLM

            Args:
                query_name (str): The name of the object to query.
                cameras (list): list of camera names, if None, use all cameras
            Returns:
                tuple: A tuple containing object points and object normals.
            """
            assert query_name in self.target_objects, f"Unknown object name: {query_name}"
            points, masks, normals = [], [], []
            if cameras is None:
                cameras = self.camera_names
            for cam in self.camera_names:
                # depth_frame = self.latest_obs[f"{cam}_depth"]
                mask_frame = self.latest_mask[cam]
                # point = convert_depth_to_pointcloud(
                #     depth_image=depth_frame,
                #     extrinsic_params=self.camera_params[cam]["extrinsic_params"],
                #     camera_intrinsics=self.camera_params[cam]["intrinsic_params"],
                #     clip_far=self.camera_params[cam]["far_near"][0],
                #     clip_near=self.camera_params[cam]["far_near"][1],
                # )
                point = self.latest_obs[f"{cam}_cloudpoint"]
                points.append(point.reshape(-1, 3))
                masks.append(
                    mask_frame.reshape(-1)
                )  # it contain the mask of different type of object
                # estimate normals using o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[-1])
                pcd.estimate_normals()  # 估计每个点的法线
                cam_normals = np.asarray(pcd.normals)
                # use lookat vector to adjust normal vectors
                flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
                cam_normals[flip_indices] *= -1
                normals.append(cam_normals)
                # break  # for test
            points = np.concatenate(points, axis=0)
            masks = np.concatenate(masks, axis=0)  # [0,101,102,201,202,0,0,0,301]
            normals = np.concatenate(normals, axis=0)
            categery_masks = (
                masks.astype(np.int32) // self.category_multiplier
            )  # [0,1,1,2,2,0,0,0,3]
            # get object points
            category_label = self.name2categerylabel[query_name]  # 1
            # objs_mask: [0,101,102,0,0,0,0,0,0]
            masks[~np.isin(categery_masks, category_label)] = 0  # [0,101,102,0,0,0,0,0,0]
            if not np.any(masks):
                # which masks == [0,0,0,0,0,0,0,0,0] if category_label == 4
                warnings.warn(f"Object {query_name} not found in the scene")
                return None
            object_instance_label = np.unique(np.mod(masks, self.category_multiplier))[
                1:
            ]  # remove the background # [1,2] which measn there are two instances of this object
            assert (
                len(object_instance_label) > 0
            ), f"Object {query_name} not found in the scene"
            objs_points = []
            objs_normals = []
            for obj_ins_id in object_instance_label:
                obj_mask = (
                    masks == obj_ins_id + self.category_multiplier * category_label
                )  # [False,True,False,False,False,False,False,False,False] for first loop
                obj_points = points[obj_mask]
                obj_normals = normals[obj_mask]
                # voxel downsample using o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(obj_points)
                pcd.normals = o3d.utility.Vector3dVector(obj_normals)
                pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
                pcd_downsampled_filted, ind = pcd_downsampled.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=1.0
                )
                obj_points = np.asarray(pcd_downsampled_filted.points)
                obj_normals = np.asarray(pcd_downsampled_filted.normals)
                objs_points.append(obj_points)
                objs_normals.append(obj_normals)
                if self.visualizer is not None:
                    self.visualizer.add_object_points(
                        np.asarray(pcd_downsampled_filted.points),
                        f"{query_name}_{obj_ins_id}",
                    )
                # save pcd to file for debug
                o3d.io.write_point_cloud(
                    f"{query_name}_{obj_ins_id}.pcd", pcd_downsampled_filted
                )
            print(f"we find {len(objs_points)} instances of {query_name}")
            return zip(objs_points, objs_normals)

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.
            ignore_grasped_obj (bool): Whether to ignore points corresponding to grasped objects.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        return self.latest_obs['front_center_cloudpoint'], None
    
    def _process_action(self, action):
        """
        Processes the given action to ensure it is within the action space.

        Args:
            action (dict): The action to process.

        Returns:
            dict: The processed action.
        """
        # clip the action
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        if type(action) == list:
            action = np.array(action)
        if self._use_old_airsim:
            action.clip([self.workspace_bounds_min[0], self.workspace_bounds_min[1], self.workspace_bounds_min[2], -1, -1, -1, -1], 
                        [self.workspace_bounds_max[0], self.workspace_bounds_max[1], self.workspace_bounds_max[2], 1, 1, 1, 1])
        return action

    def apply_action(self, action, mode="teleport", update_mask=True):
        """
        Applies the given action to the environment. 
        while the action is the target position of qudacopter in form of [x,y,z,quaternion], where quaternion is in form of [x,y,z,w],
        Args:
            action (dict): The action to apply.
        """
        assert (
            self.has_processed_first_frame
        ), "Please reset the environment first or let VLM process the first frame"
        if mode == "teleport":
            assert (
                len(action) == 7
            ), "the action should be [x,y,z,quaternion] where quaternion is in form of [x,y,z,w]"
            action = self._process_action(action)
            rospy.loginfo(f"received action {action}")
            target_pose = PoseStamped()
            target_pose.header.frame_id = self._world_frame_id
            target_pose.header.stamp = rospy.Time.now()
            target_pose.pose.position = Vector3(*action[:3])
            target_pose.pose.orientation = Quaternion(*action[3:])
            goal = MoveBaseGoal()
            goal.target_pose = target_pose
            result = self._action_client.send_goal_and_wait(goal, rospy.Duration(30))        
            if result != GoalStatus.SUCCEEDED:
                raise ValueError(f"the action {action} failed")
        elif mode == "velocity":
            # apply velocity conmand to the drone
            if len(action) == 3:
                action = np.concatenate([action, [1]])
            assert len(action) == 4, "the action should be [linear_x, linear_y, linear_z, angular_z]"
            self._cmd_pub.publish(Twist(Vector3(*action[:3]), Vector3(0,0,action[3])))
            rospy.sleep(0.2)
        if update_mask and False:
            masks = self.vlm.process_frame(self._get_rgb_frames(), verbose=True)
            if not np.any(masks):
                raise ValueError(
                    "no intrested object found in the scene, may be you should let robot turn around or change the scene or change the target object"
                )
                return None
            [
                self.latest_mask.update({cam: mask})
                for cam, mask in zip(self.camera_names, masks)
            ]
        reward = terminate = None        
        self.latest_reward = reward  # TODO
        self.latest_terminate = terminate  # TODO
        self._update_visualizer()
        return self.latest_obs, reward, terminate
    
    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        init_pose = [0,0,-1]
        
        return self.apply_action(init_pose+[0,0,0,1])
    
    def _update_visualizer(self):
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(
                ignore_robot=False, ignore_grasped_obj=False
            )
            self.visualizer.update_scene_points(points, colors)
            fig = plt.figure(figsize=(6.4 * len(self.camera_names), 4.8))
            for idx, cam in enumerate(self.camera_names):
                rgb = self.latest_obs[f"{cam}_rgb"]
                mask = np.mod(self.latest_mask[cam], 256)  # avoid overflow for color
                # trans np array to PIL image
                self.visualizer.add_mask(cam, mask)
                self.visualizer.add_rgb(cam, rgb)
                # create a subfigure for each rgb frame
                ax = fig.add_subplot(1, len(self.camera_names), idx + 1)
                ax.imshow(rgb)
                ax.imshow(mask, alpha=0.5, cmap="gray", vmin=0, vmax=255)
                ax.set_title(cam)
                # tight_layout会自动调整子图参数，使之填充整个图像区域
                plt.tight_layout()
            # trans fig to numpy array
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            self.visualizer.add_frame(data)
    
    def get_ee_pose(self):
        """
        Get the end effector pose.

        Returns:
            np.ndarray: The end effector pose.
        """
        return self.latest_obs['quad_pose']

    def get_ee_pos(self):
        """
        Get the end effector position.

        Returns:
            np.ndarray: The end effector position.
        """
        return self.latest_obs['quad_pose'][:3]
    
    def get_ee_quat(self):
        """
        Get the end effector quaternion.

        Returns:
            np.ndarray: The end effector quaternion.
        """
        return self.latest_obs['quad_pose'][3:]
    
    def get_ee_oriendation(self):
        """
        Get the end effector orientation.

        Returns:
            np.ndarray: The end effector orientation.
        """
        return self.latest_obs['quad_pose'][7:]
    
    def _reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """
        self.init_obs = None
        self.latest_obs = None
        self.latest_reward = None
        self.latest_terminate = None
        self.latest_action = None
        # scene-specific helper variables
        self.name2categerylabel = {}  # first_generation name -> label
        self.categerylabel2name = {}  # label -> first_generation name

    # def takeoff(self):
    #     self._takeoff_pub.publish(Empty())

    # def land(self):
    #     self._land_pub.publish(Empty())

    # def move(self, linear_x, linear_y, linear_z, angular_z):
    #     twist = Twist()
    #     twist.linear.x = linear_x
    #     twist.linear.y = linear_y
    #     twist.linear.z = linear_z
    #     twist.angular.z = angular_z
    #     self._cmd_vel_pub.publish(twist)

    # def get_pose(self):
    #     return latest_obs

    def camera_info_sub_front_center_callback(self, msg):
        extrinsic_params = np.array(msg.P).reshape(3, 4)
        intrinsic_params = np.array(msg.K).reshape(3, 3)
        self.camera_params = {
            "front_center": {
                "extrinsic_params": extrinsic_params,
                "intrinsic_params": intrinsic_params,
                "far_near": [3.5, 0.05],
            }
        }
        look_at = extrinsic_params[:3, :3] @ np.array([0, 0, 1])
        self.lookat_vectors['front_center'] = normalize_vector(look_at)
    
    def rgb_image_sub_front_center_callback(self, msg):
        self.latest_obs.update({
            'front_center_rgb': cv2.cvtColor(self._cvb.imgmsg_to_cv2(msg),cv2.COLOR_BGR2RGB)
        })
    
    def cloudpoint_sub_front_center_callback(self, msg:PointCloud2):
        # transform pointcloud from pointcloud_frame to world_ned
        point = np.array(list(pl2.read_points(msg, field_names = ('x', 'y', 'z'), skip_nans=True)))
        self.latest_obs.update({
            'front_center_cloudpoint': point
        })   


    def camera_info_sub_front_left_callback(self, msg):
        extrinsic_params = np.array(msg.P).reshape(3, 4)
        intrinsic_params = np.array(msg.K).reshape(3, 3)
        self.camera_params = {
            "front_left": {
                "extrinsic_params": extrinsic_params,
                "intrinsic_params": intrinsic_params,
                "far_near": [3.5, 0.05],
            }
        }
        look_at = extrinsic_params[:3, :3] @ np.array([0, 0, 1])
        self.lookat_vectors['front_left'] = normalize_vector(look_at)
    
    def rgb_image_sub_front_left_callback(self, msg):
        self.latest_obs.update({
            'front_left_rgb': cv2.cvtColor(self._cvb.imgmsg_to_cv2(msg),cv2.COLOR_BGR2RGB)
        })
    
    def cloudpoint_sub_front_left_callback(self, msg:PointCloud2):
        # transform pointcloud from pointcloud_frame to world_ned
        point = np.array(list(pl2.read_points(msg, field_names = ('x', 'y', 'z'), skip_nans=True)))
        # print(point.shape)
        self.latest_obs.update({
            'front_left_cloudpoint': point
        })   
        
            
    def camera_info_sub_front_right_callback(self, msg):
        extrinsic_params = np.array(msg.P).reshape(3, 4)
        intrinsic_params = np.array(msg.K).reshape(3, 3)
        self.camera_params = {
            "front_right": {
                "extrinsic_params": extrinsic_params,
                "intrinsic_params": intrinsic_params,
                "far_near": [3.5, 0.05],
            }
        }
        look_at = extrinsic_params[:3, :3] @ np.array([0, 0, 1])
        self.lookat_vectors['front_right'] = normalize_vector(look_at)
    
    def rgb_image_sub_front_right_callback(self, msg):
        self.latest_obs.update({
            'front_right_rgb': cv2.cvtColor(self._cvb.imgmsg_to_cv2(msg),cv2.COLOR_BGR2RGB)
        })
    
    def cloudpoint_sub_front_right_callback(self, msg:PointCloud2):
        # transform pointcloud from pointcloud_frame to world_ned
        point = np.array(list(pl2.read_points(msg, field_names = ('x', 'y', 'z'), skip_nans=True)))
        # print(point.shape)
        self.latest_obs.update({
            'front_right_cloudpoint': point
        })   
            

    def camera_info_sub_down_center_callback(self, msg):
        extrinsic_params = np.array(msg.P).reshape(3, 4)
        intrinsic_params = np.array(msg.K).reshape(3, 3)
        self.camera_params = {
            "down_center": {
                "extrinsic_params": extrinsic_params,
                "intrinsic_params": intrinsic_params,
                "far_near": [3.5, 0.05],
            }
        }
        look_at = extrinsic_params[:3, :3] @ np.array([0, 0, 1])
        self.lookat_vectors['down_center'] = normalize_vector(look_at)
    
    def rgb_image_sub_down_center_callback(self, msg):
        self.latest_obs.update({
            'down_center_rgb': cv2.cvtColor(self._cvb.imgmsg_to_cv2(msg),cv2.COLOR_BGR2RGB)
        })
    
    def cloudpoint_sub_down_center_callback(self, msg:PointCloud2):
        # transform pointcloud from pointcloud_frame to world_ned
        point = np.array(list(pl2.read_points(msg, field_names = ('x', 'y', 'z'), skip_nans=True)))
        # print(point.shape)
        self.latest_obs.update({
            'down_center_cloudpoint': point
        })   


    def camera_info_sub_rear_center_callback(self, msg):
        extrinsic_params = np.array(msg.P).reshape(3, 4)
        intrinsic_params = np.array(msg.K).reshape(3, 3)
        self.camera_params = {
            "rear_center": {
                "extrinsic_params": extrinsic_params,
                "intrinsic_params": intrinsic_params,
                "far_near": [3.5, 0.05],
            }
        }
        look_at = extrinsic_params[:3, :3] @ np.array([0, 0, 1])
        self.lookat_vectors['rear_center'] = normalize_vector(look_at)
    
    def rgb_image_sub_rear_center_callback(self, msg):
        self.latest_obs.update({
            'rear_center_rgb': cv2.cvtColor(self._cvb.imgmsg_to_cv2(msg),cv2.COLOR_BGR2RGB)
        })
    
    def cloudpoint_sub_rear_center_callback(self, msg:PointCloud2):
        # transform pointcloud from pointcloud_frame to world_ned
        point = np.array(list(pl2.read_points(msg, field_names = ('x', 'y', 'z'), skip_nans=True)))
        # print(point.shape)
        self.latest_obs.update({
            'rear_center_cloudpoint': point
        })   