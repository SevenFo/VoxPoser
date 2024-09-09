from typing import Callable, Union
import time
import scipy
import scipy.ndimage
import transforms3d.utils
import os
import rclpy.time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
import tf2_ros
import tf2_geometry_msgs
# import tf.transformations
import cv_bridge
import numpy as np
from VLMPipline.VLM import VLM
from VLMPipline.VLMM import VLMProcessWrapper
import open3d as o3d
import warnings
import cv2
from geometry_msgs.msg import (
    PoseStamped,
    Quaternion,
    Vector3,
    Twist,
    TransformStamped,
)
from actionlib_msgs.msg import GoalStatus
import transforms3d
# from std_srvs.srv import SetBool, SetBoolResponse
import matplotlib.pyplot as plt
from functools import partial
from utils import normalize_vector, bcolors, Observation, timer_decorator
from threading import Event as threading_Event
from threading import RLock as threading_Lock
import copy
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from hello_moveit_action.action import MoveAndGripper


from sensor_msgs.msg import JointState

WIDTH = 640
HEIGHT = 640


class VoxPoserROS2MPEnv:
    class MoveGroupActionClient:

        def __init__(self, node):
            self.ros_node = node
            self._action_client = ActionClient(self.ros_node, MoveAndGripper, 'move_and_gripper')

        def send_goal(self, pose_goal, gripper_state):
            goal_msg = MoveAndGripper.Goal()
            goal_msg.pose_goal = pose_goal
            goal_msg.gripper_state = gripper_state

            self._action_client.wait_for_server()

            # self._send_goal_future = self._action_client.send_goal_async(
            #     goal_msg,
            #     feedback_callback=self.feedback_callback
            # )
            # self._send_goal_future.add_done_callback(self.goal_response_callback)
            result = self._action_client.send_goal(
                goal=goal_msg
            )
            # rate = self.ros_node.create_rate(1)
            # while not self._send_goal_future.done():
            #     rate.sleep()
            #     print("wait goal")

        def goal_response_callback(self, future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.ros_node.get_logger().info('Goal rejected')
                return

            self.ros_node.get_logger().info('Goal accepted')
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(self.get_result_callback)

        def get_result_callback(self, future):
            result = future.result().result
            if result.success:
                self.ros_node.get_logger().info('Goal succeeded!')
            else:
                self.ros_node.get_logger().info('Goal failed')
            return 
            # rclpy.shutdown()

        def feedback_callback(self, feedback_msg):
            feedback = feedback_msg.feedback
            self.ros_node.get_logger().info(f'Received feedback: {feedback}')
    def __init__(
        self,
        vlmpipeline: Union[VLM, VLMProcessWrapper] = None,
        visualizer=None,
        use_old_airsim=True,
        target_objects=None,
        configs=None,
    ):
        print(f"current process ID: {os.getpid()}")
        self.target_objects = target_objects
        self._use_old_airsim = use_old_airsim
        self._cvb = cv_bridge.CvBridge()
        self.latest_action = None
        self.latest_obs = {}
        self.lookat_vectors = {}
        self.camera_params = {}
        self.vlm = vlmpipeline
        self.camera_names = [
            "front_center",
            # "front_left",
            # "front_right",
            # "down_center",
            # "rear_center",
        ]
        self.visualizer = visualizer
        self.workspace_bounds_min = np.array(configs.workspace_bounds_min)
        self.workspace_bounds_max = np.array(configs.workspace_bounds_max)
        print(self.workspace_bounds_min, self.workspace_bounds_max)
        if self.visualizer is not None:
            self.visualizer.update_bounds(
                self.workspace_bounds_min, self.workspace_bounds_max
            )
        self.configs = configs.ros_params
        if self.configs is not None:
            self.camera_names = [item["name"] for item in self.configs.cameras]
        self._lock = threading_Lock()
        self.init_ros()
        self.init_task()

    def init_task(self):
        self.descriptions = [
            "fly around a distance above the table",
            "From under the table, cross the past to the 100cm in front of the table, then fly to the top 100cm above the table",
            "fly to the table",
            "go to the table",
        ]
        self.target_objects = (
            [
                "pumpkin",
                "house",
                "apple",
                "Stone lion statue",
                "windmill",
            ]
            if self.target_objects is None
            else self.target_objects
        )
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

    def init_ros(self):
        rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
        self.ros_node = rclpy.create_node('voxposer')
        self.ros_logger = self.ros_node.get_logger()
        self._tf_buffer = tf2_ros.Buffer(rclpy.time.Duration(seconds=100))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer,self.ros_node, spin_thread=True)
        self._update_ee_pose_timer = self.ros_node.create_timer(1, self.update_ee_pose)
        
        # 初始化参数
        self._world_frame_id = self.configs.world_frame_id
        self._drone_frame_id = self.configs.drone_origin_frame_id

        self._cloudpoint_subs = {}
        self._rgb_image_subs = {}
        self._rgb_camera_info_subs = {}

        self.ros_logger.info(
            f"wait transform from {self.configs.cameras[0]['rgb_frame_id']} to {self._world_frame_id}"
        )
        rate = self.ros_node.create_rate(1)
        while not self._tf_buffer.can_transform(
            self._world_frame_id,
            self.configs.cameras[0]["rgb_frame_id"],
            rclpy.time.Time(),
            rclpy.time.Duration(),
        ):
            self.ros_logger.info(
                f"can not transform from {self.configs.cameras[0]['rgb_frame_id']} to {self._world_frame_id}"
            )
            rate.sleep()
        self._tf_buffer.lookup_transform(
            self._world_frame_id, self.configs.cameras[0]["rgb_frame_id"], rclpy.time.Time()
        )
        self.ros_logger.info(
            f"get transform from {self.configs.cameras[0]['rgb_frame_id']} to {self._world_frame_id}"
        )
        self._cloudpoint_sub_events = []
        for item in self.configs.cameras:
            camera_name = item["name"]
            self._cloudpoint_sub_events.append(threading_Event())
            self._cloudpoint_subs[camera_name] = self.ros_node.create_subscription(
                PointCloud2,
                item["cloudpoint_topic_name"],
                partial(
                    self.cloudpoint_sub_calllback_template,
                    key=f"{camera_name}_cloudpoint",
                    event=self._cloudpoint_sub_events[-1],
                ),
                qos_profile=10
            )
            self._rgb_image_subs[camera_name] = self.ros_node.create_subscription(
                Image,
                item["rgb_topic_name"],
                partial(self.rgb_image_sub_callback_template, key=f"{camera_name}_rgb"),
                qos_profile=10
            )
            self._rgb_camera_info_subs[camera_name] = self.ros_node.create_subscription(
                CameraInfo,
                item["rgb_camera_info_topic_name"],
                partial(
                    self.camera_info_sub_callback_template, camera_name=camera_name
                ),
                qos_profile=10
            )
        # joint_state updater
        self._joint_state_sub = self.ros_node.create_subscription(JointState,"/isaac_joint_states", self.joint_state_callback, qos_profile=10)
        # self._cmd_pub = self.ros_node.create_publisher(
        #     Twist, "/airsim_node/vel_cmd_world_frame",  queue_size=1
        # )
        # self._odom_sub = self.ros_node.create_subscription(
        #     self.configs.drone_odom_topic_name, Odometry, self.odom_sub_callback,qos_profile=10
        # )

        # self._action_client = actionlib.SimpleActionClient(
        #     self.configs.action_client_name, MoveBaseAction
        # )

        # if self.configs.reset_service_name != "":
        #     self.ros_logger.info(f"wait for {self.configs.reset_service_name} service")
        #     rospy.wait_for_service(self.configs.reset_service_name, timeout=10)
        #     self.ros_logger.info(
        #         f"reset env: {rospy.ServiceProxy(self.configs.reset_service_name, SetBool)(SetBool._request_class(True)).message}"
        #     )
        #     self.reset_service_proxy = rospy.ServiceProxy(
        #         self.configs.reset_service_name, SetBool
        #     )
        
        self._action_client = VoxPoserROS2MPEnv.MoveGroupActionClient(self.ros_node)

    def get_object_names(self):
        return self.target_objects

    def update_ee_pose(self):
        ee_frame = "panda_hand"
        if not self._tf_buffer.can_transform(ee_frame,self._world_frame_id,rclpy.time.Time(),timeout=rclpy.time.Duration(seconds=0.5)):
            self.ros_logger.info(f"cant trans from {self._world_frame_id} to {ee_frame}")
            return 
        trans = self._tf_buffer.lookup_transform(self._world_frame_id,ee_frame,rclpy.time.Time(),timeout=rclpy.time.Duration(seconds=0.5))
        pose = trans.transform
        with self._lock:
            self.latest_obs.update(
                {
                    "quad_pose": np.array(
                        [
                            pose.translation.x,
                            pose.translation.y,
                            pose.translation.z,
                            pose.rotation.x,
                            pose.rotation.y,
                            pose.rotation.z,
                            pose.rotation.w,
                        ]
                        + list(
                            transforms3d.euler.quat2euler(
                                [
                                    pose.rotation.w,
                                    pose.rotation.x,
                                    pose.rotation.y,
                                    pose.rotation.z,
                                ]
                            )
                        )
                    )
                }
            )

    def _snap_obs(self):
        # deepcopy the latest_obs
        with self._lock:
            self.snaped_obs = copy.deepcopy(self.latest_obs)

    def _get_rgb_frames(self):
        self._snap_obs()
        rgb_frames = {}  # in c w h
        for cam in self.camera_names:
            while (
                f"{cam}_rgb" not in self.snaped_obs
                or f"{cam}_cloudpoint" not in self.snaped_obs
            ):
                # raise ValueError(f"no {cam}_rgb found")
                self.ros_logger.info(f"waiting for {cam}_rgb image and {cam}_cloudpoint")
                self._snap_obs()
                self.ros_node.create_rate(2).sleep()
            rgb_frames[cam] = self.snaped_obs[f"{cam}_rgb"].transpose([2, 0, 1])
        frames = np.stack(list(rgb_frames.values()), axis=0)
        # pcd = self.snaped_obs[f"{self.camera_names[0]}_cloudpoint"]
        # np.savetxt("snaped_cloudpoint.txt", pcd)
        # plt.imshow((self.snaped_obs[f"{self.camera_names[0]}_rgb"]).astype(np.uint8))
        # plt.savefig("snaped_rgb.png")

        return frames

    def reset(self):
        descriptions = self.descriptions[0]
        self.init_obs = self.latest_obs
        self.latest_mask = {}
        self.target_objects_labels = list(range(len(self.target_objects)))
        frames = self._get_rgb_frames()  # use snaped obs
        # masks = self.vlm.process_first_frame(
        #     self.target_objects, frames, verbose=True, owlv2_threshold=0.2
        # )
        masks = self.vlm.process_first_frame(frames)
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
            point = self.snaped_obs[
                f"{cam}_cloudpoint"
            ]  # use naped obs to sync cloudpoint with rgb (also mask)
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
        object_instance_label = np.unique(
            np.mod(masks, self.category_multiplier)
        )[
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
            # plt.imshow(
            #     (obj_mask.astype(np.uint8) * np.array([255], dtype=np.uint8)).reshape(
            #         HEIGHT, WIDTH
            #     )
            # )
            # plt.savefig(f"{query_name}_{obj_ins_id}_mask.png")
            obj_points = points[obj_mask]
            obj_normals = normals[obj_mask]
            # remove nan from point
            nan_mask = ~np.isnan(obj_points).any(axis=1)
            obj_points = obj_points[nan_mask]
            obj_normals = obj_normals[nan_mask]
            # np.savetxt(
            #     f"{query_name}_{obj_ins_id}_cloudpoint_all.txt", np.asarray(points)
            # )
            # np.savetxt(
            #     f"{query_name}_{obj_ins_id}_cloudpoint_np.txt", np.asarray(obj_points)
            # )
            # voxel downsample using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points)
            pcd.normals = o3d.utility.Vector3dVector(obj_normals)
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
            pcd_downsampled_filted, ind = pcd_downsampled.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=1.0
            )
            obj_points_dsf = np.asarray(pcd_downsampled_filted.points)
            obj_normals_dsf = np.asarray(pcd_downsampled_filted.normals)
            objs_points.append(obj_points_dsf)
            objs_normals.append(obj_normals_dsf)
            if self.visualizer is not None:
                self.visualizer.add_object_points(
                    obj_normals_dsf,
                    f"{query_name}_{obj_ins_id}",
                )
            # save pcd to file for debug
            # o3d.io.write_point_cloud(
            #     f"{query_name}_{obj_ins_id}.pcd", pcd_downsampled_filted
            # )
            # np.savetxt(
            #     f"{query_name}_{obj_ins_id}_cloudpoint.txt", np.asarray(obj_points)
            # )
            # np.savetxt(
            #     f"{query_name}_{obj_ins_id}_cloudpoint_ds.txt",
            #     np.asarray(pcd_downsampled.points),
            # )
            # np.savetxt(
            #     f"{query_name}_{obj_ins_id}_cloudpoint_dsf.txt",
            #     np.asarray(pcd_downsampled_filted.points),
            # )
            assert len(obj_points) > 0, f"no points found for {query_name}_{obj_ins_id}"
        print(f"we find {len(objs_points)} instances of {query_name}")
        return list(zip(objs_points, objs_normals))

    def get_scene_3d_obs(
        self, ignore_robot=False, ignore_grasped_obj=False, do_gaussian_filter=False
    ):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.
            ignore_grasped_obj (bool): Whether to ignore points corresponding to grasped objects.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        while f"{self.configs.cameras[0]['name']}_cloudpoint" not in self.latest_obs:
            self.ros_logger.info("waiting for cloudpoint")
            self.ros_node.create_rate(2).sleep()
        if do_gaussian_filter:
            # filte test
            image_shape = self.latest_obs[
                f"{self.configs.cameras[0]['name']}_rgb"
            ].shape
            x, y, z = np.split(
                self.latest_obs[f"{self.configs.cameras[0]['name']}_cloudpoint"],
                3,
                axis=1,
            )
            y = y.reshape(image_shape[0], image_shape[1])
            self.ros_logger.info(f"std of y: {np.std(y)}")
            y = scipy.ndimage.gaussian_filter(y, sigma=0.01, radius=1)
            return np.concatenate([x, y.reshape(-1, 1), z], axis=1), None
        return self.latest_obs[f"{self.configs.cameras[0]['name']}_cloudpoint"], None

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
            action.clip(
                [
                    self.workspace_bounds_min[0],
                    self.workspace_bounds_min[1],
                    self.workspace_bounds_min[2],
                    -1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    self.workspace_bounds_max[0],
                    self.workspace_bounds_max[1],
                    self.workspace_bounds_max[2],
                    1,
                    1,
                    1,
                    1,
                ],
            )
        return action

    def apply_action(self, action, mode="teleport", update_mask=True,speed = None):
                # action 的前7个元素是EE位姿，第8个元素是抓手状态
        ee_pose = action[:7]
        gripper_state = action[7]

        # 将EE位姿转换为关节位置并发布
        self.move_to_pose(ee_pose)
        
        # 发布抓手状态
        self.set_gripper_state(gripper_state)
        
        if update_mask:
            # masks = self.vlm.process_frame(self._get_rgb_frames(), verbose=True)
            masks = self.vlm.process_frame(self._get_rgb_frames())  # snap obs
            if not np.any(masks):
                warnings.warn(
                    "no intrested object found in the scene, may be you should let robot turn around or change the scene or change the target object"
                )
                return None
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

    def move_to_pose(self, pose):
        print("send goal")
        self._action_client.send_goal(pose,0)
        print("goal executed")
        
    def set_gripper_state(self, state):
        return 
        joint_state = JointState()
        joint_state.name = ['panda_finger_joint1', 'panda_finger_joint2']
        joint_state.position = [0.04 if state else 0.0] * 2
        joint_state.velocity = [0.0] * 2
        joint_state.effort = [0.0] * 2
        self.joint_command_publisher.publish(joint_state)
        
    def reset_to_default_pose(self):
        return 
        self.joint_command_publisher.publish(self.default_joint_state)

    def pose_to_joint_positions(self, pose):
        return 
        # 实现逆运动学计算
        plan = self.move_group.plan()
        if plan.joint_trajectory.points:
            return [point.positions for point in plan.joint_trajectory.points]
        else:
            return None

    def publish_joint_positions(self, joint_positions):
        return 
        joint_state = JointState()
        joint_state.name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        joint_state.position = joint_positions
        joint_state.velocity = [0.0] * len(joint_positions)
        joint_state.effort = [0.0] * len(joint_positions)
        self.joint_command_publisher.publish(joint_state)
        
        
    def _transform_traj(self, traj, trans):
        """
        Applies a transformation to a trajectory.

        Args:
            traj (list): The input trajectory as a list of points.
            trans (Transform): The transformation to apply.

        Returns:
            list: The transformed trajectory as a list of points.
        """
        # 提取平移和旋转信息
        translation = trans.transform.translation
        rotation = trans.transform.rotation

        # 将旋转四元数转换为旋转矩阵
        quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
        rotation_matrix = tf.transformations.quaternion_matrix(quaternion)[0:3, 0:3]

        # 构建4x4变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = rotation_matrix
        transform_matrix[0:3, 3] = [translation.x, translation.y, translation.z]

        # 对traj中的每个路径点应用变换
        transformed_traj = []
        for point in traj:
            # 将路径点转换为齐次坐标形式
            point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
            # 进行矩阵乘法
            transformed_point_homogeneous = np.dot(transform_matrix, point_homogeneous)
            # 提取转换后的点坐标
            transformed_point = transformed_point_homogeneous[0:3]
            transformed_traj.append(transformed_point.tolist())

        return transformed_traj

    def _set_traj_to_params_server(self, traj):
        num_wp = len(traj)
        prefix = "drone_0_ego_planner_node/fsm"
        self.ros_logger.info(f"params server prefix: {prefix}")
        rospy.set_param(f"{prefix}/waypoint_num", num_wp)
        self.ros_logger.info(f"set {prefix}/waypoint_num: {num_wp}")
        [
            (rospy.set_param(f"{prefix}/waypoint{i}_{d}", value[j]), self.ros_logger.info(f"set {prefix}/waypoint{i}_{d}: {value[j]}"))
            for i, value in enumerate(traj)
            for j, d in enumerate(["x", "y", "z"])
        ]
        rospy.set_param(
            f"{prefix}/renew_goals", 1
        )  # set flag to weak up lower controller


    @timer_decorator
    def _update_visualizer(self):
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(
                ignore_robot=False, ignore_grasped_obj=False
            )
            self.visualizer.update_scene_points(points, colors)

            ## TODO code below takes too much time, use coroutine/concurrent.futures to speed up
            # fig = plt.figure(figsize=(6.4 * len(self.camera_names), 4.8))
            # for idx, cam in enumerate(self.camera_names):
            #     rgb = self.snaped_obs[f"{cam}_rgb"]
            #     mask = np.mod(self.latest_mask[cam], 256)  # avoid overflow for color
            #     # trans np array to PIL image
            #     self.visualizer.add_mask(cam, mask)
            #     self.visualizer.add_rgb(cam, rgb)
            #     # create a subfigure for each rgb frame
            #     ax = fig.add_subplot(1, len(self.camera_names), idx + 1)
            #     ax.imshow(rgb)
            #     ax.imshow(mask, alpha=0.5, cmap="gray", vmin=0, vmax=255)
            #     ax.set_title(cam)
            #     # tight_layout会自动调整子图参数，使之填充整个图像区域
            #     plt.tight_layout()
            # # trans fig to numpy array
            # fig.canvas.draw()
            # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # self.visualizer.add_frame(data)

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return self.init_obs["gripper_open"]

    def get_ee_pose(self):
        """
        Get the end effector pose.

        Returns:
            np.ndarray: The end effector pose.
        """
        return self.latest_obs["quad_pose"]

    def get_ee_pos(self):
        """
        Get the end effector position.

        Returns:
            np.ndarray: The end effector position.
        """
        # self.ros_logger.info(f"get_ee_pos: {self.latest_obs['quad_pose'][:3]}")
        print(f'get ee pos: {self.latest_obs["quad_pose"][:3]}')
        return self.latest_obs["quad_pose"][:3]

    def get_ee_quat(self):
        """
        Get the end effector quaternion.

        Returns:
            np.ndarray: The end effector quaternion.
        """
        return self.latest_obs["quad_pose"][3:7]

    def get_ee_oriendation(self):
        """
        Get the end effector orientation.

        Returns:
            np.ndarray: The end effector orientation.
        """
        return self.latest_obs["quad_pose"][7:]

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

    def cloudpoint_sub_calllback_template(
        self, msg: PointCloud2, key: str, event: threading_Event
    ):
        # 除那12字节外，最后4字节填了个固定值kPointCloudComponentFourMagic(1)，有的文章把这分量称为强度或者反射值r
        point = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3)[:, 0:3]
        # np.savetxt(f"{key}_pcl_origin.xyz",point,delimiter=",")
        if msg.header.frame_id != self._world_frame_id:
            trans: TransformStamped = self._tf_buffer.lookup_transform(
                self._world_frame_id, msg.header.frame_id, rclpy.time.Time()
            )
            rotation_matrix = transforms3d.quaternions.quat2mat(
                [
                    trans.transform.rotation.w,
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                ]
            )
            translation = np.array(
                [
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z,
                ]
            ).reshape(1, 3)
            point = point @ rotation_matrix.T + translation
        # np.savetxt(f"{key}_pcl.xyz",point,delimiter=",")
        with self._lock:
            self.latest_obs.update({f"{key}": point})

    def rgb_image_sub_callback_template(self, msg: Image, key: str):
        # # rgb_data = cv2.cvtColor(self._cvb.imgmsg_to_cv2(msg), cv2.COLOR_BGR2RGB)
        rgb_data = self._cvb.imgmsg_to_cv2(msg) ## RGB?? as encoding is passthrough ??
        cv2.imwrite(f"{key}_image.png",rgb_data)
        with self._lock:
            self.latest_obs.update({f"{key}": rgb_data})

    def camera_info_sub_callback_template(self, msg: CameraInfo, camera_name: str):
        extrinsic_params = np.array(msg.p).reshape(3, 4)
        intrinsic_params = np.array(msg.k).reshape(3, 3)
        self.camera_params = {
            f"{camera_name}": {
                "extrinsic_params": extrinsic_params,
                "intrinsic_params": intrinsic_params,
                "far_near": [3.5, 0.05],
            }
        }
        look_at = extrinsic_params[:3, :3] @ np.array([0, 0, 1])
        self.lookat_vectors[f"{camera_name}"] = normalize_vector(look_at)

    def joint_state_callback(self, msg: JointState):
        joint_names = msg.name
        joint_states_raw = {}
        for idx, name in enumerate(joint_names):
            joint_states_raw.update({
                f"{name}":{
                    f"position": msg.position[idx],
                    f"velocity": msg.position[idx],
                    f"effort": msg.position[idx],
                }
            })
        gripper_open = 0 if all(np.array(msg.position[-2:]) < 0.035) else 1        
        with self._lock:
            self.latest_obs.update({
                "joint_states_raw":joint_states_raw,
                "gripper_open": gripper_open
            })
    
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
