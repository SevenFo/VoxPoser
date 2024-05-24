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


class DummyEnv:
    def __init__(self, vlmpipeline: VLM = None, visualizer=None):
        print(f"current process ID: {os.getpid()}")
        self.latest_obs = {}
        self.lookat_vectors = {}
        self.camera_params = {}
        self._cvb = cv_bridge.CvBridge()
        self.vlm = vlmpipeline
        self.camera_names = [
            "front_center",
            "front_left",
            "front_right",
            "down_center",
            "rear_center",
        ]
        self.visualizer = visualizer
        self.workspace_bounds_min = np.array([0, 0, 0])
        self.workspace_bounds_max = np.array([5, 5, 3])
        if self.visualizer is not None:
            self.visualizer.update_bounds(
                self.workspace_bounds_min, self.workspace_bounds_max
            )

        self.init_task()

    def init_task(self):
        self.descriptions = [
            "fly around a distance above the table",
            "From under the table, cross the past to the 100cm in front of the table, then fly to the top 100cm above the table",
            "fly to the table",
            "go to the table",
        ]
        self.target_objects = [
            "pumpkin",
            "house",
            "apple",
            "Stone lion statue",
            "windmill",
        ]
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

    def reset(self):
        return self.descriptions[0], self.latest_obs

    def get_object_names(self):
        return self.target_objects

    def get_ee_pos(self):
        return np.array([0, 0, 0])

    def get_3d_obs_by_name_by_vlm(self, objname):
        return []

    def reset_to_default_pose(self):
        pass

    def get_scene_3d_obs(self):
        return None

    def get_last_gripper_action(self):
        return None

    def get_ee_quat(self):
        return np.array([0, 0, 0, 1])
