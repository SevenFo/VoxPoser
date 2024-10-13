from typing import Callable, Union
import time
import scipy
import scipy.ndimage
import transforms3d.utils
import rclpy
import rclpy.clock
import rclpy.time
from rclpy.node import Node
import os
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from VLMPipline.VLM import VLM
from VLMPipline.VLMM import VLMProcessWrapper
from LMP_MA import StateError
from geometry_msgs.msg import (
    PoseStamped,
    Quaternion,
    Vector3,
    Twist,
    TransformStamped,
)
from rclpy.action import ActionClient
# from move_base_msgs.action import MoveBase
from nav2_msgs.action import NavigateToPose, NavigateThroughPoses
from action_msgs.msg import GoalStatus
import transforms3d
from std_srvs.srv import SetBool
import matplotlib.pyplot as plt
from functools import partial
from utils import normalize_vector, bcolors, Observation, timer_decorator
from threading import Event as threading_Event
from threading import RLock as threading_Lock
import copy
import cv_bridge
import numpy as np
import open3d as o3d
import warnings
import tf2_geometry_msgs
import tf2_ros
import base64
import requests

WIDTH = 640
HEIGHT = 640


class VoxPoserROS2DroneEnv(Node):
    def __init__(
        self,
        vlmpipeline: Union[VLM, VLMProcessWrapper] = None,
        visualizer=None,
        use_old_airsim=True,
        target_objects=None,
        configs=None,
    ):
        super().__init__('voxposer_ros2_drone_env')
        print(f"current process ID: {os.getpid()}")
        self.target_objects = target_objects
        self._use_old_airsim = use_old_airsim
        self._cvb = cv_bridge.CvBridge()
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
        self._detect_memory ={}
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
        self._tf_buffer = tf2_ros.Buffer(cache_time=rclpy.time.Duration(seconds=10))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # 初始化参数
        self._world_frame_id = self.configs.world_frame_id
        self._drone_frame_id = self.configs.drone_origin_frame_id
        self._drone_base_link_frame_id = self.configs.drone_baselink_frame_id

        self._cloudpoint_subs = {}
        self._rgb_image_subs = {}
        self._rgb_camera_info_subs = {}

        self.get_logger().info(
            f"wait transform from {self.configs.cameras[0]['rgb_frame_id']} to {self._world_frame_id}"
        )
        while not self._tf_buffer.can_transform(
            self._world_frame_id,
            self.configs.cameras[0]["rgb_frame_id"],
            rclpy.time.Time(),
            rclpy.duration.Duration(seconds=0.1),
        ):
            self.get_logger().info(
                f"can not transform from {self.configs.cameras[0]['rgb_frame_id']} to {self._world_frame_id}, at {self.get_clock().now()}"
            )
            rclpy.spin_once(self)
        self._tf_buffer.lookup_transform(
            self._world_frame_id, self.configs.cameras[0]["rgb_frame_id"], rclpy.time.Time()
        )
        self.get_logger().info(
            f"get transform from {self.configs.cameras[0]['rgb_frame_id']} to {self._world_frame_id}"
        )
        self._cloudpoint_sub_events = []
        for item in self.configs.cameras:
            camera_name = item["name"]
            self._cloudpoint_sub_events.append(threading_Event())
            self._cloudpoint_subs[camera_name] = self.create_subscription(
                PointCloud2,
                item["cloudpoint_topic_name"],
                partial(
                    self.cloudpoint_sub_calllback_template,
                    key=f"{camera_name}_cloudpoint",
                    event=self._cloudpoint_sub_events[-1],
                ),
                10
            )
            self._rgb_image_subs[camera_name] = self.create_subscription(
                Image,
                item["rgb_topic_name"],
                partial(self.rgb_image_sub_callback_template, key=f"{camera_name}_rgb"),
                10
            )
            self._rgb_camera_info_subs[camera_name] = self.create_subscription(
                CameraInfo,
                item["rgb_camera_info_topic_name"],
                partial(
                    self.camera_info_sub_callback_template, camera_name=camera_name
                ),
                10
            )

        self._cmd_pub = self.create_publisher(
            Twist, "/airsim_node/vel_cmd_world_frame", 1
        )
        # self._odom_sub = self.create_subscription(
        #     Odometry, self.configs.drone_odom_topic_name, self.odom_sub_callback, 10
        # ) 
        self._drone_pose_sub = self.create_subscription(
            PoseStamped, self.configs.drone_pose_topic_name, self.drone_pose_sub_callback, rclpy.qos.qos_profile_sensor_data
        )

        # TODO
        self._action_client = ActionClient(
            self, NavigateToPose, self.configs.action_client_name
        )

        if self.configs.reset_service_name != "":
            self.get_logger().info(f"wait for {self.configs.reset_service_name} service")
            self.reset_service_proxy = self.create_client(SetBool, self.configs.reset_service_name)
            while not self.reset_service_proxy.wait_for_service(timeout_sec=10.0):
                self.get_logger().info(
                    f"reset env: {self.reset_service_proxy.call(SetBoolRequest(data=True)).message}"
                )

    def get_object_names(self):
        return self.target_objects

    def drone_pose_sub_callback(self, msg: PoseStamped):
        pose = msg
        if msg.header.frame_id != self._world_frame_id:
            # transform odom from odom_frame to world_frame
            trans = self._tf_buffer.lookup_transform(
                self._world_frame_id, msg.header.frame_id, rclpy.time.Time()
            )
            pose = tf2_geometry_msgs.do_transform_pose_stamped(pose, trans).pose
        else:
            pose = pose.pose
        with self._lock:
            self.latest_obs.update(
                {
                    "quad_pose": np.array(
                        [
                            pose.position.x,
                            pose.position.y,
                            pose.position.z,
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        ]
                        + list(
                            transforms3d.euler.quat2euler(
                                [
                                    pose.orientation.w,
                                    pose.orientation.x,
                                    pose.orientation.y,
                                    pose.orientation.z,
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
                self.get_logger().info(f"waiting for {cam}_rgb image and {cam}_cloudpoint")
                self._snap_obs()
                # rclpy.spin_once(self, timeout_sec=0.01)
                time.sleep(1)
            rgb_frames[cam] = self.snaped_obs[f"{cam}_rgb"].transpose([2, 0, 1])
        frames = np.stack(list(rgb_frames.values()), axis=0)
        return frames

    def reset(self):
        descriptions = self.descriptions[0]
        self.init_obs = self.latest_obs
        self.latest_mask = {}
        self.target_objects_labels = list(range(len(self.target_objects)))
        frames = self._get_rgb_frames()  # use snaped obs
        masks = self.vlm.process_first_frame(frames)
        if not np.any(masks):
            raise StateError(
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
    
    def _request_process_sole_frame(self, labels, data_array):
        url = "http://127.0.0.1:5000/process_sole_frame"
        data_bytes = data_array.tobytes()
        data_base64 = base64.b64encode(data_bytes).decode("utf-8")
        
        data = {
            "label": labels,
            "data": data_base64,
            "shape": data_array.shape
        }
        
        resp = requests.post(url, json=data)
        
        if resp.status_code == 200:
            result = resp.json()
            return np.array(result).astype(np.uint32)
        else:
            print(f"Request process_sole_frame failed with status code {resp.status_code}")
            return None
    
    def get_3d_obs_by_name_by_vlm(self, query_name, cameras=None, is_sole = False):
        """
        Retrieves 3D point cloud observations and normals of an object by its name by VLM

        Args:
            query_name (str): The name of the object to query.
            cameras (list): list of camera names, if None, use all cameras
        Returns:
            tuple: A tuple containing object points and object normals.
        """
        assert query_name in self.target_objects or is_sole, f"Unknown object name: {query_name}"
        points, masks, normals = [], [], []
        cameras = self.camera_names if cameras is None else cameras
        
        if is_sole:
            rgb_frames = self._get_rgb_frames()
            masks_ = self._request_process_sole_frame([query_name], rgb_frames)
            # if masks_ is None: 
            #     if query_name not in self._detect_memory:
            #         return None
            #     else:
            #         self.get_logger().warning(f"use detect memory to fetch {query_name}")
            #         return self._detect_memory[query_name]
        else:
            masks_ = [self.latest_mask[cam] for cam in cameras]
            
        for idx, cam in enumerate(cameras):
            mask_frame = masks_[idx]
            point = self.snaped_obs[f"{cam}_cloudpoint"]  # use naped obs to sync cloudpoint with rgb (also mask)
            points.append(point.reshape(-1, 3))
            masks.append(mask_frame.reshape(-1))  # it contain the mask of different type of object
            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[-1])
            pcd.estimate_normals()  # 估计每个点的法线
            cam_normals = np.asarray(pcd.normals)
            # use lookat vector to adjust normal vectors
            flip_indices = np.dot(cam_normals, self.lookat_vectors[cam]) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)
        points = np.concatenate(points, axis=0)
        masks = np.concatenate(masks, axis=0)  # [0,101,102,201,202,0,0,0,301]
        normals = np.concatenate(normals, axis=0)
        categery_masks = (
            masks.astype(np.int32) // self.category_multiplier
        )  # [0,1,1,2,2,0,0,0,3]
        # get object points
        category_label = self.name2categerylabel[query_name] if not is_sole else 1  # 1 
        # objs_mask: [0,101,102,0,0,0,0,0,0]
        masks[~np.isin(categery_masks, category_label)] = 0  # [0,101,102,0,0,0,0,0,0]
        if not np.any(masks):
            if query_name not in self._detect_memory:
                # which masks == [0,0,0,0,0,0,0,0,0] if category_label == 4
                warnings.warn(f"Object {query_name} not found in the scene")
                return None
            else:
                self.get_logger().warning(f"use detect memory to fetch {query_name}")
                return self._detect_memory[query_name]
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
            obj_points = points[obj_mask]
            obj_normals = normals[obj_mask]
            # remove nan from point
            nan_mask = ~np.isnan(obj_points).any(axis=1)
            obj_points = obj_points[nan_mask]
            obj_normals = obj_normals[nan_mask]
            # voxel downsample using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points)
            pcd.normals = o3d.utility.Vector3dVector(obj_normals)
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
            pcd_downsampled_filted, ind = pcd_downsampled.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=1.0
            )
            obj_points_dsf = np.asarray(pcd_downsampled_filted.points)
            np.savetxt(f"obj_points_dsf_{obj_ins_id}.txt",obj_points_dsf,delimiter=",")
            obj_normals_dsf = np.asarray(pcd_downsampled_filted.normals)
            objs_points.append(obj_points_dsf)
            objs_normals.append(obj_normals_dsf)
            if self.visualizer is not None:
                self.visualizer.add_object_points(
                    obj_normals_dsf,
                    f"{query_name}_{obj_ins_id}",
                )
            assert len(obj_points) > 0, f"no points found for {query_name}_{obj_ins_id}"
        result = list(zip(objs_points, objs_normals))
        self._detect_memory[query_name] = result
        print(f"we find {len(objs_points)} instances of {query_name}")
        return result

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
        merged_cloud = o3d.geometry.PointCloud()
        while f"{self.configs.cameras[0]['name']}_cloudpoint" not in self.latest_obs:
            self.get_logger().info("waiting for cloudpoint")
            time.sleep(1)
        self._snap_obs()
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
            self.get_logger().info(f"std of y: {np.std(y)}")
            y = scipy.ndimage.gaussian_filter(y, sigma=0.01, radius=1)
            return np.concatenate([x, y.reshape(-1, 1), z], axis=1), None
        scene_points = []
        merged_cloud.points = o3d.utility.Vector3dVector(self.snaped_obs[f"{self.configs.cameras[0]['name']}_cloudpoint"])
        for camera in self.camera_names[1:]:
            current_cloud = o3d.geometry.PointCloud()
            current_cloud.points = o3d.utility.Vector3dVector(self.snaped_obs[f"{camera}_cloudpoint"])
            icp_result = o3d.pipelines.registration.registration_icp(
                current_cloud, merged_cloud, max_correspondence_distance=0.02,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            current_cloud.transform(icp_result.transformation)
            merged_cloud += current_cloud
        return np.array(merged_cloud.points), None
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
        if mode == "teleport" or mode == "pose":
            assert (
                len(action) == 7
            ), "the action should be [x,y,z,quaternion] where quaternion is in form of [x,y,z,w]"
            action = self._process_action(action)
            self.get_logger().info(f"received action {action}")
            target_pose = PoseStamped()
            target_pose.header.frame_id = self._world_frame_id
            target_pose.header.stamp = self.get_clock().now().to_msg()
            target_pose.pose.position = Vector3(*action[:3])
            target_pose.pose.orientation = Quaternion(*action[3:])
            trans = self._tf_buffer.lookup_transform(
                self._drone_frame_id,
                target_pose.header.frame_id,
                rclpy.time.Time(),  # ????
            )
            target_pose = tf2_geometry_msgs.do_transform_pose(target_pose, trans)
            goal = MoveBase.Goal()
            goal.target_pose = target_pose
            self.get_logger().info_once("waiting for drone to reach the target position")
            self._action_client.wait_for_server()
            future = self._action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            result = future.result()
            if result.status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().error("Failed to reach the target position")
            self.get_logger().info("Successfully reached the target position")

        elif mode == "velocity":
            # apply velocity conmand to the drone
            if len(action) == 3:
                action = np.concatenate([action, [1]])
            assert (
                len(action) == 4
            ), "the action should be [linear_x, linear_y, linear_z, angular_z]"
            self._cmd_pub.publish(Twist(Vector3(*action[:3]), Vector3(0, 0, action[3])))
            time.sleep(0.1) if not update_mask else None
        elif mode == "traj":
            traj = [a[0] for a in action]
            self.get_logger().info(f"received traj in {self._world_frame_id} {traj}")
            trans = self._tf_buffer.lookup_transform(
                self._drone_frame_id,
                self._world_frame_id,
                rclpy.time.Time(),  # ????
            )
            traj = self._transform_traj(traj, trans)
            self.get_logger().info(f"trans traj to {self._drone_frame_id} {traj}")
            self._set_traj_to_params_server(traj)
            self.get_logger().info_once("waiting for drone to finish traj")
            # avoid blocking
            # while rospy.get_param("/fsm/drone_state") != 1:
            #     rospy.sleep(0.1)
        if update_mask:
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

    def is_finished(self):
        return self.get_parameter("/fsm/drone_state").get_parameter_value().integer_value == 1

    def train_traj_to_drone_origin_frame(self, traj):
        """
        Transforms a trajectory from the world frame to the drone's origin frame.

        Args:
            traj (list): The input trajectory as a list of points.

        Returns:
            list: The transformed trajectory as a list of points.
        """
        trans = self._tf_buffer.lookup_transform(
            self._drone_frame_id,
            self._world_frame_id,
            rclpy.time.Time(),  # ????
        )
        self.get_logger().info(f"Trans from {self._world_frame_id} to {self._drone_frame_id}: {trans}")
        # trans = self._tf_buffer.lookup_transform(
        #     self._world_frame_id,
        #     self._drone_frame_id,
        #     rclpy.time.Time(),  # ????
        # )
        # self.get_logger().info(f"Trans from {self._drone_frame_id} to {self._world_frame_id}: {trans}")
        traj = self._transform_traj(traj, trans)
        return traj

    def _transform_traj(self, traj, trans):
        """
        Applies a transformation to a trajectory.

        Args:
            traj (list): The input trajectory as a list of points.
            trans (TransformStamped): The transformation to apply.

        Returns:
            list: The transformed trajectory as a list of points.
        """
        # 提取平移和旋转信息
        translation = trans.transform.translation
        rotation = trans.transform.rotation
        # rotation_matrix = transforms3d.quaternions.quat2mat(
        #     [
        #         trans.transform.rotation.w,
        #         trans.transform.rotation.x,
        #         trans.transform.rotation.y,
        #         trans.transform.rotation.z,
        #     ]
        # )
        # translation = np.array(
        #     [
        #         trans.transform.translation.x,
        #         trans.transform.translation.y,
        #         trans.transform.translation.z,
        #     ]
        # ).reshape(1, 3)
        # point = point @ rotation_matrix.T + translation
        # 将旋转四元数转换为旋转矩阵
        quaternion = [rotation.w, rotation.x, rotation.y, rotation.z]
        rotation_matrix = transforms3d.quaternions.quat2mat(quaternion)
         
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
        self.get_logger().info(f"params server prefix: {prefix}")
        self.set_parameters([
            rclpy.parameter.Parameter(f"{prefix}/waypoint_num", rclpy.Parameter.Type.INTEGER, num_wp)
        ])
        self.get_logger().info(f"set {prefix}/waypoint_num: {num_wp}")
        for i, value in enumerate(traj):
            for j, d in enumerate(["x", "y", "z"]):
                self.set_parameters([
                    rclpy.parameter.Parameter(f"{prefix}/waypoint{i}_{d}", rclpy.Parameter.Type.DOUBLE, value[j])
                ])
                self.get_logger().info(f"set {prefix}/waypoint{i}_{d}: {value[j]}")
        self.set_parameters([
            rclpy.parameter.Parameter(f"{prefix}/renew_goals", rclpy.Parameter.Type.INTEGER, 1)
        ])  # set flag to wake up lower controller

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        init_pose = [0, 0, -1]
        return self.apply_action(init_pose + [0, 0, 0, 1])

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
        # rospy.loginfo(f"get_ee_pos: {self.latest_obs['quad_pose'][:3]}")
        return self.latest_obs["quad_pose"][:3]

    def get_ee_quat(self):
        """
        Get the end effector quaternion.

        Returns:
            np.ndarray: The end effector quaternion.
        """
        return self.latest_obs["quad_pose"][3:]

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

    def cloudpoint_sub_calllback_template(self, msg: PointCloud2, key: str, event: threading_Event):
        point = self._pointcloud2_to_array(msg)
        # np.savetxt(f"scene_points_origin_{key}.xyz",point.reshape(-1, 3),delimiter=',')
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
            # # y +90 x+90 dynamic
            # rotation_x_neg_90 = transforms3d.euler.euler2mat(+np.pi / 2, -np.pi / 2, 0, 'ryxz')
            # point = point @ rotation_x_neg_90.T
            # np.savetxt(f"scene_points_{key}.xyz",point.reshape(-1, 3),delimiter=',')
        with self._lock:
            self.latest_obs[key] = point

    def rgb_image_sub_callback_template(self, msg: Image, key: str):
        # # rgb_data = cv2.cvtColor(self._cvb.imgmsg_to_cv2(msg), cv2.COLOR_BGR2RGB)
        with self._lock:
            self.latest_obs[key] = self._cvb.imgmsg_to_cv2(msg)

    def camera_info_sub_callback_template(self, msg: CameraInfo, camera_name: str):
        self.camera_params[camera_name] = {
            "K": np.array(msg.k).reshape((3, 3)),
            "D": np.array(msg.d),
            "R": np.array(msg.r).reshape((3, 3)),
            "P": np.array(msg.p).reshape((3, 4)),
            "far_near": [3.5, 0.05],
        }
        old = normalize_vector(self.camera_params[camera_name]["P"][:3,:3] @ np.array([0,0,1]))
        # new = normalize_vector(np.dot(np.linalg.inv(self.camera_params[camera_name]["R"]), np.array([0, 0, 1])))
#         self.get_logger().info(f'\
# old: {normalize_vector(self.camera_params[camera_name]["P"][:3,:3] @ np.array([0,0,1]))}, \
# new: {normalize_vector(np.dot(np.linalg.inv(self.camera_params[camera_name]["R"]), np.array([0, 0, 1])))}'
#         )
        self.lookat_vectors[camera_name] = old
    def _pointcloud2_to_array(self, cloud_msg: PointCloud2):
        dtype_list = self._get_pointcloud2_dtype(cloud_msg)
        cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)
        cloud_arr = cloud_arr.view(np.float32).reshape(cloud_msg.height, cloud_msg.width, -1)
        return cloud_arr[:, :, :3].reshape((-1, 3))

    def _get_pointcloud2_dtype(self, cloud_msg: PointCloud2):
        dtype_list = []
        for field in cloud_msg.fields:
            if field.datatype == PointField.FLOAT32:
                dtype_list.append((field.name, np.float32))
            elif field.datatype == PointField.FLOAT64:
                dtype_list.append((field.name, np.float64))
            elif field.datatype == PointField.UINT8:
                dtype_list.append((field.name, np.uint8))
            elif field.datatype == PointField.INT8:
                dtype_list.append((field.name, np.int8))
            elif field.datatype == PointField.UINT16:
                dtype_list.append((field.name, np.uint16))
            elif field.datatype == PointField.INT16:
                dtype_list.append((field.name, np.int16))
            elif field.datatype == PointField.UINT32:
                dtype_list.append((field.name, np.uint32))
            elif field.datatype == PointField.INT32:
                dtype_list.append((field.name, np.int32))
            else:
                raise ValueError(f"Unknown PointField datatype [{field.datatype}]")
        return dtype_list
    
    @timer_decorator
    def _update_visualizer(self):
        if self.visualizer:
            points, colors = self.get_scene_3d_obs(
                ignore_robot=False, ignore_grasped_obj=False
            )
            np.savetxt("scene_points.xyz",points.reshape(-1, 3),delimiter=',')
            self.visualizer.update_scene_points(points, colors)


if __name__ == "__main__":

    def main(args=None):
        rclpy.init(args=args)
        node = VoxPoserROS2DroneEnv()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
        
    main()