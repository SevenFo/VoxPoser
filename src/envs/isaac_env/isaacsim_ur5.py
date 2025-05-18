# env_isaac_sim.py
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
# (And modifications by user for Isaac Sim Core integration)
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import inspect
from typing import Any, Dict, Optional, List, Tuple, Union
import os
import torch
import logging
import base64
import requests  # requests is synchronous
import numpy as np
import open3d as o3d
import time
import functools

import carb
from pxr import Gf, UsdGeom

from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.app import get_app
import omni.kit.async_engine
from omni.isaac.core.objects import Camera as IsaacCoreCamera  # Renamed to avoid clash
from omni.isaac.core.utils.rotations import (
    quat_to_euler_angles,
)  # For convenience if needed

# Isaac Lab imports (some can be reused, others are replaced)
from omni.isaac.lab.utils.math import (
    subtract_frame_transforms,
    quat_rotate,
    euler_xyz_from_quat,
    quat_from_matrix,
    quat_mul,
    quat_from_angle_axis,
    quat_inv,
    matrix_from_quat,
    axis_angle_from_quat,  # Added for DIK
    transform_points,  # Reusable
)
from omni.isaac.lab.sensors.camera.utils import (  # Reusable utilities
    create_pointcloud_from_depth,
)

# Your provided scripts (assuming they are in the same directory or accessible via python path)
# For this example, I'll assume RobotController, ConfigManager are available.
# If they are in a subfolder, adjust imports: e.g., from .robot_controller import RobotController
try:
    from robot_controller import RobotController  # Referenced script
    from config_manager import ConfigManager  # Referenced script
    from ur5_control.ur5 import UR5  # Referenced by RobotController
    from dik_controller.differential_ik import DifferentialIKController  # Referenced
    from dik_controller.differential_ik_cfg import (
        DifferentialIKControllerCfg,
    )  # Referenced
except ImportError as e:
    print(
        f"Failed to import RobotController or related components: {e}. Ensure they are in PYTHONPATH."
    )
    print(
        "You might need to create dummy files or adjust imports if these are not fully defined yet."
    )

    # Define dummy classes if imports fail, to allow the rest of the structure to be parsed
    class ConfigManager:
        def __init__(self, path):
            pass

        def get(self, key, default=None):
            return default

    class RobotController:
        def __init__(self, cfg_mgr):
            pass

        def initialize(self):
            self.my_world = None  # Placeholder
            self.my_ur = None  # Placeholder
            self.dik_controller = None  # Placeholder
            self.robot_config = {}
            self.logger = logging.getLogger("DummyRobotController")
            return True

        def _set_initial_pose(self):
            pass

    class UR5:
        pass

    class DifferentialIKController:
        pass


# Reusable Observation class and decorator from your EnvIsaacLab
def omni_async_to_sync(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        coro = f(*args, **kwargs)
        task = omni.kit.async_engine.run_coroutine(coro)
        while not task.done():
            app_interface = get_app()
            if app_interface:
                app_interface.update()
            else:
                time.sleep(0.001)
        if task.exception():
            original_exception = task.exception()
            raise type(original_exception)(str(original_exception)).with_traceback(
                original_exception.__traceback__
            )
        return task.result()

    return wrapper


async_to_sync = omni_async_to_sync


class Observation:
    def __init__(self):
        self._data: Dict[str, Any] = {}

    def update(self, other: Dict[str, Any]) -> None:
        for k, v in other.items():
            if isinstance(v, dict):
                if k not in self._data or not isinstance(self._data[k], Observation):
                    self._data[k] = Observation()
                if isinstance(self._data[k], Observation):
                    self._data[k].update(v)
                else:
                    new_obs = Observation()
                    new_obs.update(v)
                    self._data[k] = new_obs
            else:
                if inspect.iscoroutine(v):
                    self._data[k] = omni.kit.async_engine.run_coroutine(v)
                else:
                    self._data[k] = v

    def _resolve_value(self, value: Any) -> Any:
        if isinstance(value, asyncio.Task):
            while not value.done():
                app_interface = get_app()
                if app_interface:
                    app_interface.update()
                else:
                    time.sleep(0.001)
            if value.exception():
                original_exception = value.exception()
                raise type(original_exception)(str(original_exception)).with_traceback(
                    original_exception.__traceback__
                )
            return value.result()
        return value

    def _getitem_internal(self, key):
        if key not in self._data:
            raise KeyError(f"Key '{key}' not found in Observation data.")
        value = self._data[key]
        value = self._resolve_value(value)
        if isinstance(value, dict):
            obs_wrapper = Observation()
            obs_wrapper.update(value)
            self._data[key] = obs_wrapper
            return obs_wrapper
        if isinstance(self._data[key], asyncio.Task) and not isinstance(
            value, asyncio.Task
        ):
            self._data[key] = value
        return value

    def __getitem__(self, key):
        return self._getitem_internal(key)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            obs = Observation()
            obs.update(value)
            self._data[key] = obs
        elif inspect.iscoroutine(value):
            self._data[key] = omni.kit.async_engine.run_coroutine(value)
        else:
            self._data[key] = value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key in self._data

    def to_dict(self) -> Dict[str, Any]:
        resolved_dict = {}
        for k_loop, v_initial_loop in self.items():
            if isinstance(v_initial_loop, Observation):
                resolved_dict[k_loop] = v_initial_loop.to_dict()
            else:
                resolved_dict[k_loop] = v_initial_loop
        return resolved_dict

    def items(self):
        for key_ in list(self._data.keys()):
            yield key_, self[key_]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvIsaacSim:
    def __init__(self, cfg: dict, visualizer=None):  # visualizer not used yet
        self.cfg = cfg
        self.robot_cfg = self.cfg.get(
            "robot_controller_cfg", {}
        )  # Config for RobotController
        self.env_cfg = self.cfg.get(
            "environment_cfg", {}
        )  # Config for EnvIsaacSim specific things

        # Initialize RobotController (which sets up World, Robot, Task)
        # Assuming robot_controller_cfg.yaml contains paths and robot details
        # cfg_path = self.robot_cfg.get("config_file_path", "path/to/your/robot_controller_config.yaml")
        # self.config_manager = ConfigManager(cfg_path)
        # Hardcoding some config for RobotController for now, adapt with your ConfigManager
        # This part needs to align with how your RobotController expects its config
        robot_controller_configs = {
            "robot": {
                "usd_path": self.robot_cfg.get(
                    "robot_usd_path", "/Isaac/Robots/UR5e/ur5e.usd"
                ),
                "end_effector_prim_name": self.robot_cfg.get(
                    "end_effector_prim_name", "tool0"
                ),  # Crucial for Jacobian
                "initial_joints": self.robot_cfg.get(
                    "initial_joint_angles_deg", [0, -90, 90, -90, -90, 0]
                ),  # Degrees
                "base_position": self.robot_cfg.get(
                    "base_position", [0, 0, 0.0]
                ),  # Check Z height
                "use_gripper": self.robot_cfg.get(
                    "use_gripper", True
                ),  # Assuming a gripper is used
                "gripper_joint_names": self.robot_cfg.get(
                    "gripper_joint_names",
                    [
                        "left_outer_knuckle_joint",
                        "right_outer_knuckle_joint",
                        "left_inner_finger_joint",
                        "right_inner_finger_joint",
                    ],
                ),  # Example names
                # Define open/closed states for the gripper (radians)
                "gripper_open_angles_rad": self.robot_cfg.get(
                    "gripper_open_angles_rad", [0.0, 0.0, 0.0, 0.0]
                ),  # Fully open
                "gripper_closed_angles_rad": self.robot_cfg.get(
                    "gripper_closed_angles_rad", [0.8, 0.8, 0.8, 0.8]
                ),  # Fully closed
                "joint_names_without_gripper": self.robot_cfg.get(
                    "arm_joint_names",
                    [
                        "shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint",
                    ],
                ),
            },
            "task": {
                "scene_path": self.robot_cfg.get(
                    "scene_usd_path", None
                ),  # e.g. "/path/to/lunar_base.usd"
                "target_position": self.robot_cfg.get(
                    "default_target_cube_pos", [0.5, 0, 0.4]
                ),
            },
            # "rtde": {"enabled": False} # Assuming RTDE is not used for sim
        }
        self.robot_controller = RobotController(
            configs=robot_controller_configs
        )  # Pass the sub-config
        if not self.robot_controller.initialize():
            raise RuntimeError("Failed to initialize RobotController for EnvIsaacSim.")

        self.world = self.robot_controller.my_world
        self.robot_articulation: UR5 = (
            self.robot_controller.my_ur
        )  # This is your UR5 instance
        self.dik_controller: DifferentialIKController = (
            self.robot_controller.dik_controller
        )

        self._set_camera_view_to_target_object()
        self.world.step(render=True)  # Initial step

        self.target_objects = self.env_cfg.get("scene_target_objects", [])
        self.cameras_to_use_names = self.env_cfg.get(
            "cameras_to_use", ["wrist", "front"]
        )  # Names like "wrist", "front"
        self.cameras_core: Dict[
            str, IsaacCoreCamera
        ] = {}  # Stores IsaacCoreCamera objects
        self.camera_info: Dict[str, Any] = {}  # Stores matrices, lookat vectors
        self._setup_cameras()

        self.category_multiplier = self.env_cfg.get("vlm_category_multiplier", 100)
        self.name2categerylabel = {
            name: i for i, name in enumerate(self.target_objects, start=1)
        }
        self.categerylabel2name = {
            i: name for i, name in enumerate(self.target_objects, start=1)
        }

        self.device = "cpu"  # Or torch.device("cuda:0") if using GPU tensors

        self.latest_obs = Observation()
        self.init_obs = Observation()

        self.workspace_bounds_min = np.array(
            self.env_cfg.get("workspace_bounds_min", [-0.1, -0.5, 0.0])
        )
        self.workspace_bounds_max = np.array(
            self.env_cfg.get("workspace_bounds_max", [1.0, 0.5, 1.0])
        )

        self.use_vlm = self.env_cfg.get("use_vlm", True)
        self.latest_action_raw_input: Optional[np.ndarray] = (
            None  # Store the input to apply_action
        )
        self.latest_reward: Optional[float] = None  # Not implemented yet
        self.latest_terminate: Optional[bool] = False  # Not implemented yet

        self.gripper_is_open: bool = (
            True  # Initial assumption, will be updated in reset
        )
        self.gripper_open_angles_rad = np.array(
            self.robot_cfg["robot"]["gripper_open_angles_rad"]
        )
        self.gripper_closed_angles_rad = np.array(
            self.robot_cfg["robot"]["gripper_closed_angles_rad"]
        )

        # Store initial EE pose from robot's default state after initialization
        # Note: RobotController already calls _set_initial_pose which sets joint angles
        self.world.step(render=True)  # ensure physics is propagated
        pos, quat_wxyz = self.robot_articulation.get_end_effector_pose(
            body_name=self.robot_cfg["robot"]["end_effector_prim_name"]
        )
        self.initial_ee_pos = pos.copy()
        self.initial_ee_quat_wxyz = quat_wxyz.copy()  # w,x,y,z
        self.initial_gripper_is_open = True  # Assume starts open after reset

        logger.info(
            f"EnvIsaacSim initialized. Target objects: {self.target_objects}, VLM: {self.use_vlm}"
        )
        logger.info(f"Robot EE: {self.robot_cfg['robot']['end_effector_prim_name']}")
        logger.info(
            f"Initial EE pose: {self.initial_ee_pos}, {self.initial_ee_quat_wxyz}"
        )

    def _setup_cameras(self):
        """Creates and configures IsaacCoreCamera instances."""
        camera_configs = self.env_cfg.get(
            "camera_configurations",
            {
                "wrist": {
                    "prim_path": "/World/Robot/UR5e/tool0/wrist_camera",  # Example path, adjust to your robot USD
                    "position_offset": [
                        0.0,
                        0.05,
                        -0.06,
                    ],  # Relative to EE link (tool0)
                    "orientation_quat_wxyz_offset": quat_mul(  # optical Z-fwd, Y-down from ROS Y-fwd, Z-up
                        quat_from_angle_axis(
                            torch.tensor(torch.pi), torch.tensor([1.0, 0, 0])
                        ),  # 180 deg around X
                        quat_from_angle_axis(
                            torch.tensor(-0.5 * torch.pi), torch.tensor([0, 0, 1.0])
                        ),  # -90 deg around Z
                    )
                    .cpu()
                    .numpy()
                    .tolist(),  # [w,x,y,z] for Gf.Quat
                    "resolution": [256, 256],  # H, W
                    "attach_to_ee": True,
                },
                "front": {
                    "prim_path": "/World/front_camera",
                    "position": [1.6, 0.0, 0.6],  # World position
                    "orientation_quat_wxyz": quat_mul(  # Look at origin from (1.6,0,0.6)
                        quat_from_angle_axis(
                            torch.tensor(torch.pi / 3.0), torch.tensor([0, 1, 0])
                        ),
                        quat_from_angle_axis(
                            torch.tensor(0.5 * torch.pi), torch.tensor([0, 0, 1.0])
                        ),
                    )
                    .cpu()
                    .numpy()
                    .tolist(),  # [w,x,y,z]
                    "resolution": [480, 640],
                    "attach_to_ee": False,
                },
            },
        )

        for cam_name in self.cameras_to_use_names:
            if cam_name not in camera_configs:
                logger.warning(
                    f"Configuration for camera '{cam_name}' not found. Skipping."
                )
                continue

            cfg = camera_configs[cam_name]

            # Convert wxyz to Gf.Quat which is (real, i, j, k) -> (w, x, y, z)
            if "orientation_quat_wxyz_offset" in cfg:
                # Offset is relative to parent link's orientation
                orientation_gf = Gf.Quatd(
                    float(cfg["orientation_quat_wxyz_offset"][0]),
                    float(cfg["orientation_quat_wxyz_offset"][1]),
                    float(cfg["orientation_quat_wxyz_offset"][2]),
                    float(cfg["orientation_quat_wxyz_offset"][3]),
                )
            elif "orientation_quat_wxyz" in cfg:
                orientation_gf = Gf.Quatd(
                    float(cfg["orientation_quat_wxyz"][0]),
                    float(cfg["orientation_quat_wxyz"][1]),
                    float(cfg["orientation_quat_wxyz"][2]),
                    float(cfg["orientation_quat_wxyz"][3]),
                )
            else:  # Default orientation (identity)
                orientation_gf = Gf.Quatd(1, 0, 0, 0)

            if cfg["attach_to_ee"]:
                # For EE camera, position is an offset. We'll update its pose dynamically.
                # The prim_path should be under the end-effector link.
                # We create the camera object but will set its local transform before each capture.
                ee_prim_path = (
                    self.robot_articulation.prim_path
                    + "/"
                    + self.robot_cfg["robot"]["end_effector_prim_name"]
                )
                camera_prim_path = cfg[
                    "prim_path"
                ]  # Assumes it's already part of USD or we create it

                self.cameras_core[cam_name] = IsaacCoreCamera(
                    prim_path=camera_prim_path,  # e.g., /World/Robot/UR5e/tool0/wrist_camera
                    # Position and orientation will be set relative to EE link
                    resolution=cfg["resolution"],
                )
                # Store offset to apply later
                self.camera_info[f"{cam_name}_pos_offset_ee"] = np.array(
                    cfg["position_offset"]
                )
                self.camera_info[f"{cam_name}_quat_offset_ee_gf"] = orientation_gf

            else:  # Fixed world camera
                self.cameras_core[cam_name] = IsaacCoreCamera(
                    prim_path=cfg["prim_path"],
                    position=np.array(cfg["position"]),
                    orientation=np.array(
                        [orientation_gf.GetReal(), *orientation_gf.GetImaginary()]
                    ),  # w,x,y,z np array
                    resolution=cfg["resolution"],
                )

            self.cameras_core[cam_name].initialize()
            self.cameras_core[cam_name].set_clipping_range(
                0.01, 100.0
            )  # sensible defaults
            logger.info(
                f"Camera '{cam_name}' setup at {cfg['prim_path']} with resolution {cfg['resolution']}."
            )
        self.world.step(render=True)  # Step to ensure cameras are registered

    def _set_camera_view_to_target_object(self):
        # Same as EnvIsaacLab
        viewport = get_active_viewport()
        if viewport:
            set_camera_view(
                eye=self.env_cfg.get("default_viewport_eye", [1.6, 0.0, 0.6]),
                target=self.env_cfg.get("default_viewport_target", [0.0, 0.0, 0.0]),
                viewport_api=viewport,
            )
        else:
            logger.warning("No active viewport found to set camera view.")

    def _request_vlm_service(
        self, endpoint: str, labels: List[str], data_array: np.ndarray
    ) -> Optional[np.ndarray]:
        # Same as EnvIsaacLab
        url = f"{self.env_cfg.get('vlm_service_base_url', 'http://127.0.0.1:5000')}/{endpoint}"
        data_bytes = data_array.tobytes()
        data_base64 = base64.b64encode(data_bytes).decode("utf-8")
        payload = {"label": labels, "data": data_base64, "shape": data_array.shape}
        try:
            resp = requests.post(
                url, json=payload, timeout=self.env_cfg.get("vlm_request_timeout", 10)
            )
            resp.raise_for_status()
            result = resp.json()
            return np.array(result).astype(np.uint32)
        except requests.exceptions.RequestException as e:
            logger.error(f"VLM request to {url} failed: {e}")
            return None
        except ValueError as e:
            logger.error(f"Failed to decode VLM response from {url}: {e}")
            return None

    def _request_process_first_frame(
        self, labels: List[str], data_array: np.ndarray
    ) -> Optional[np.ndarray]:
        return self._request_vlm_service("process_first_frame", labels, data_array)

    def _request_process_frame(self, cam_obs: Dict[str, Any]) -> Dict[str, Any]:
        # Same as EnvIsaacLab
        rgbs_list = [
            cam_obs[f"{cam}_rgb"]
            for cam in self.cameras_to_use_names
            if f"{cam}_rgb" in cam_obs
        ]
        if not rgbs_list:
            logger.warning("No RGB data found in cam_obs for VLM processing.")
            cam_obs["masks"] = None
            return cam_obs
        rgbs = np.stack(rgbs_list, axis=0)
        masks = self._request_vlm_service("process_frame", self.target_objects, rgbs)
        cam_obs["masks"] = masks
        return cam_obs

    def _get_robot_obs(self) -> Dict[str, Any]:
        """Helper to get current robot state."""
        robot_obs_dict = {}

        # EE Pose (world frame)
        # Ensure end_effector_prim_name is correctly configured for your UR5 instance
        ee_prim_name = self.robot_cfg["robot"]["end_effector_prim_name"]
        ee_pos_w, ee_quat_wxyz_w = self.robot_articulation.get_end_effector_pose(
            body_name=ee_prim_name
        )
        robot_obs_dict["ee_pos_w"] = ee_pos_w.copy()
        robot_obs_dict["ee_quat_w"] = ee_quat_wxyz_w.copy()  # w,x,y,z

        # Joint positions (radians)
        joint_pos_rad = (
            self.robot_articulation.get_joint_positions()
        )  # Gets all, including gripper
        arm_joint_names = self.robot_cfg["robot"]["joint_names_without_gripper"]
        arm_joint_indices = [
            self.robot_articulation.get_dof_index(name) for name in arm_joint_names
        ]
        robot_obs_dict["joint_pos_rad"] = joint_pos_rad[arm_joint_indices].copy()

        # Gripper state
        gripper_joint_names = self.robot_cfg["robot"]["gripper_joint_names"]
        if gripper_joint_names:
            gripper_joint_indices = [
                self.robot_articulation.get_dof_index(name)
                for name in gripper_joint_names
            ]
            current_gripper_angles = joint_pos_rad[gripper_joint_indices]
            # Simple check: if closer to open_angles than closed_angles (average distance)
            dist_to_open = np.linalg.norm(
                current_gripper_angles - self.gripper_open_angles_rad
            )
            dist_to_closed = np.linalg.norm(
                current_gripper_angles - self.gripper_closed_angles_rad
            )
            self.gripper_is_open = dist_to_open < dist_to_closed
        robot_obs_dict["gripper_open"] = 1.0 if self.gripper_is_open else 0.0

        return robot_obs_dict

    @async_to_sync
    async def get_camera_observation(self) -> Dict[str, Any]:
        """Gets observations from all configured IsaacCoreCamera instances."""
        logger.debug("Starting camera observation (Isaac Sim Core)")
        camera_data_dict: Dict[str, Any] = {}

        # Ensure physics is up-to-date for correct camera positioning
        await omni.kit.async_engine.yield_async()  # Yield to allow other tasks
        self.world.render()  # This is synchronous, steps physics and renders
        await get_app().next_update_async()  # Crucial for rendering to texture

        # Get current EE world pose *once* if any camera is attached to it
        ee_world_pos, ee_world_quat_wxyz = None, None
        if any(
            self.env_cfg["camera_configurations"][name].get("attach_to_ee", False)
            for name in self.cameras_to_use_names
        ):
            ee_prim_name = self.robot_cfg["robot"]["end_effector_prim_name"]
            ee_world_pos, ee_world_quat_wxyz = (
                self.robot_articulation.get_end_effector_pose(body_name=ee_prim_name)
            )

        for cam_name, cam_core in self.cameras_core.items():
            cam_cfg = self.env_cfg["camera_configurations"][cam_name]

            current_cam_pos_w = None
            current_cam_quat_wxyz_w = None

            if cam_cfg.get("attach_to_ee", False):
                if ee_world_pos is None:  # Should have been fetched above
                    logger.error(
                        f"EE pose not available for EE-attached camera {cam_name}"
                    )
                    continue

                # Transform offset from EE frame to World frame
                offset_pos_ee = self.camera_info[
                    f"{cam_name}_pos_offset_ee"
                ]  # Stored as np.array
                offset_quat_ee_gf = self.camera_info[
                    f"{cam_name}_quat_offset_ee_gf"
                ]  # Stored as Gf.Quatd

                # Convert Gf.Quatd to numpy [w,x,y,z]
                offset_quat_ee_wxyz_np = np.array(
                    [offset_quat_ee_gf.GetReal(), *offset_quat_ee_gf.GetImaginary()]
                )

                # Apply transform: T_world_cam = T_world_ee * T_ee_cam
                # Position: p_cam_w = p_ee_w + R_ee_w * p_cam_ee
                current_cam_pos_w = (
                    ee_world_pos
                    + quat_rotate(
                        torch.from_numpy(ee_world_quat_wxyz_w).float().unsqueeze(0),
                        torch.from_numpy(offset_pos_ee).float().unsqueeze(0),
                    )
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                # Orientation: q_cam_w = q_ee_w * q_cam_ee
                current_cam_quat_wxyz_w = (
                    quat_mul(
                        torch.from_numpy(ee_world_quat_wxyz_w).float(),
                        torch.from_numpy(offset_quat_ee_wxyz_np).float(),
                    )
                    .cpu()
                    .numpy()
                )

                cam_core.set_world_pose(
                    position=current_cam_pos_w, orientation=current_cam_quat_wxyz_w
                )
            else:  # Fixed camera, pose already set, but get it for info
                current_cam_pos_w, current_cam_quat_wxyz_w = cam_core.get_world_pose()

            # Data acquisition
            # These get_() methods in IsaacCoreCamera typically trigger a capture if data is stale
            rgb = cam_core.get_rgba()[:, :, :3]  # HxWxC, remove alpha
            depth = cam_core.get_depth(type="distance_to_image_plane")  # HxW

            # Intrinsics and Extrinsics
            # IsaacCoreCamera.get_intrinsic_matrix() is K: [[fx,0,cx],[0,fy,cy],[0,0,1]]
            # IsaacCoreCamera.get_extrinsic_matrix() is T_camera_world (view matrix)
            # We need T_world_camera for point cloud transformation
            intrinsic_m = cam_core.get_intrinsics_matrix()  # 3x3 np array
            extrinsic_m_cam_world = (
                cam_core.get_extrinsic_matrix()
            )  # 4x4 np array, camera_T_world

            # T_world_camera = T_camera_world_inv
            # For SE(3) matrix inv(T) = [R.T, -R.T*p]
            # Or, simpler: use current_cam_pos_w, current_cam_quat_wxyz_w to build T_world_camera
            # (since get_extrinsic_matrix might be delayed if pose was just set)

            # Build T_world_camera (OpenGL convention from Isaac Sim camera)
            # Position is current_cam_pos_w
            # Rotation from current_cam_quat_wxyz_w (which is w,x,y,z)
            # Matrix_from_quat expects [N, 4] tensor (x,y,z,w) but Gf.Quat is (w, [x,y,z])
            # For safety, use the components
            q = Gf.Quatd(
                float(current_cam_quat_wxyz_w[0]),
                float(current_cam_quat_wxyz_w[1]),
                float(current_cam_quat_wxyz_w[2]),
                float(current_cam_quat_wxyz_w[3]),
            )
            # This is rotation matrix for world_R_camera_body
            # OpenGL camera looks down -Z. Optical frame looks down +Z.
            # Isaac Sim cameras are OpenGL: X right, Y up, Z backward (out of screen)
            R_world_camBody = q.GetMatrix().ExtractRotationMatrix()  # Gf.Matrix3d
            R_world_camBody_np = np.array(R_world_camBody).reshape(3, 3)

            T_world_camBody = np.eye(4)
            T_world_camBody[:3, :3] = R_world_camBody_np
            T_world_camBody[:3, 3] = current_cam_pos_w

            # For point cloud: depth is from OpenGL camera (Z backward).
            # create_pointcloud_from_depth expects depth from optical (Z forward).
            # The default orientation in create_pointcloud_from_depth (identity) assumes optical.
            # If depth is -ve Z, then we might need to flip depth or adjust transform.
            # Isaac Sim's "distance_to_image_plane" is positive.
            # The key is the `orientation` param in `create_pointcloud_from_depth`.
            # If Isaac Sim depth is already optical-like (positive Z is distance), use identity for `orientation`.
            # If Isaac Sim depth is OpenGL (positive Z is *into* screen, meaning negative distance from sensor if sensor points forward),
            # then we need a 180deg rotation around X for `orientation` to map to optical.
            # Given "distance_to_image_plane" is positive, and optical frame has Z forward,
            # we assume it's compatible. A common convention is Y-down in optical.
            # Let's use the same q_x_180 as EnvIsaacLab, assuming depth values are distances.
            q_optical_to_body_wxyz = (
                quat_from_angle_axis(torch.tensor(torch.pi), torch.tensor([1.0, 0, 0]))
                .cpu()
                .numpy()
            )  # w,x,y,z

            pointcloud_cam_body = create_pointcloud_from_depth(
                intrinsic_matrix=torch.from_numpy(intrinsic_m).float(),
                depth=torch.from_numpy(depth).float(),  # HxW
                orientation=torch.from_numpy(
                    q_optical_to_body_wxyz
                ).float(),  # Transform from optical to this "cam_body" frame
                keep_invalid=True,  # HxWx3
            )  # Output is in a frame "cam_body" which is q_optical_to_body_wxyz rotated from optical frame.

            # Transform pointcloud_cam_body to world
            # T_world_pointcloud = T_world_camBody * T_camBody_points
            # Points are already in "cam_body", so just transform by T_world_camBody
            pointcloud_w = transform_points(
                pointcloud_cam_body.view(-1, 3),  # Nx3
                position=torch.from_numpy(current_cam_pos_w).float(),
                orientation=torch.from_numpy(
                    current_cam_quat_wxyz_w
                ).float(),  # w,x,y,z
            ).view(depth.shape[0], depth.shape[1], 3)  # HxWx3

            # lookat_w = R_world_camBody @ [0,0,-1] (OpenGL -Z view vector in world)
            lookat_w = R_world_camBody_np @ np.array([0, 0, -1.0])

            self.camera_info[f"{cam_name}_extrinsic_matrice"] = (
                T_world_camBody  # T_world_cameraBody
            )
            self.camera_info[f"{cam_name}_intrinsic_matrices"] = intrinsic_m
            self.camera_info[f"{cam_name}_lookat"] = lookat_w

            camera_data_dict[f"{cam_name}_rgb"] = np.transpose(rgb, (2, 0, 1))  # CxHxW
            camera_data_dict[f"{cam_name}_depth"] = np.expand_dims(
                depth, axis=-1
            )  # HxWx1
            camera_data_dict[f"{cam_name}_pointcloud"] = (
                pointcloud_w.cpu().numpy()
            )  # HxWx3

        logger.debug("Camera observation (Isaac Sim Core) finished.")
        return camera_data_dict

    def _process_obs(self, do_vlm: bool = True) -> Dict[str, Any]:
        """Helper to combine robot and camera observations, optionally with VLM."""
        processed_data_dict = self._get_robot_obs()

        raw_camera_data = (
            self.get_camera_observation()
        )  # This is now sync due to decorator

        if self.use_vlm and do_vlm:
            cam_data_for_vlm = (
                raw_camera_data.copy() if isinstance(raw_camera_data, dict) else {}
            )
            processed_camera_data_with_vlm = self._request_process_frame(
                cam_data_for_vlm
            )
            processed_data_dict["camera"] = processed_camera_data_with_vlm
        else:
            if isinstance(raw_camera_data, dict) and "masks" not in raw_camera_data:
                raw_camera_data["masks"] = None
            processed_data_dict["camera"] = raw_camera_data
        return processed_data_dict

    def reset(self) -> Observation:
        logger.info("Resetting environment (Isaac Sim Core)...")
        self.robot_controller._set_initial_pose()  # Resets robot arm to initial joint angles

        # Reset gripper to open state explicitly
        self.robot_articulation.set_gripper_positions(self.gripper_open_angles_rad)
        self.gripper_is_open = True
        for _ in range(5):  # Few steps for gripper to settle
            self.world.step(render=True)

        # TODO: Reset other dynamic scene objects if necessary (e.g., cube position)
        # Example: if there's a manipulable cube:
        # target_cube_prim = self.world.scene.get_object("target_cube_name_in_scene")
        # if target_cube_prim:
        #     target_cube_prim.set_world_pose(position=np.array([0.5,0,0.1]), orientation=np.array([1,0,0,0]))

        obs_dict_raw = self._process_obs(
            do_vlm=True
        )  # Gets robot, camera, and VLM for first frame

        self.init_obs.update(obs_dict_raw)
        self.latest_obs.update(obs_dict_raw)

        if self.use_vlm and (
            not obs_dict_raw.get("camera", {}).get("masks") is not None
        ):
            logger.warning(
                "VLM failed to process first frame on reset or no masks returned."
            )

        self.latest_action_raw_input = None
        self.latest_reward = None
        self.latest_terminate = False
        logger.info("Environment reset complete (Isaac Sim Core).")
        return self.latest_obs

    def apply_action(
        self, action_values_xyz_wxyz_gripper: Union[np.ndarray, List[float]]
    ) -> Tuple[Observation, Optional[float], Optional[bool]]:
        # action_values: [target_ee_pos_x, y, z, target_ee_quat_w, x, y, z, gripper_command (-1 close, 0 hold, 1 open)]
        action_np = np.asarray(action_values_xyz_wxyz_gripper, dtype=np.float32)
        self.latest_action_raw_input = action_np.copy()

        target_ee_pos_w = (
            torch.from_numpy(action_np[:3]).float().to(self.device).unsqueeze(0)
        )
        target_ee_quat_wxyz_w = (
            torch.from_numpy(action_np[3:7]).float().to(self.device).unsqueeze(0)
        )  # w,x,y,z
        gripper_command = action_np[7]  # -1 close, 0 hold, 1 open

        ee_prim_name = self.robot_cfg["robot"]["end_effector_prim_name"]

        # --- Arm Movement Phase (using DIK) ---
        # The DIK controller is relative, so we need to compute deltas in a loop
        movement_timeout_s = self.env_cfg.get("movement_timeout_s", 5.0)
        pos_threshold_abs = self.env_cfg.get("position_threshold_abs", 0.01)  # meters
        orient_threshold_abs = self.env_cfg.get(
            "orientation_threshold_abs_rad", 0.05
        )  # radians
        control_freq = self.env_cfg.get("control_frequency_hz", 10.0)  # Hz
        dt = 1.0 / control_freq
        max_steps = int(movement_timeout_s * control_freq)

        logger.debug(
            f"Applying action. Target EE: pos={action_np[:3]}, quat={action_np[3:7]}. Gripper: {gripper_command}"
        )

        for step_idx in range(max_steps):
            current_ee_pos_w_np, current_ee_quat_wxyz_w_np = (
                self.robot_articulation.get_end_effector_pose(body_name=ee_prim_name)
            )
            current_ee_pos_w = (
                torch.from_numpy(current_ee_pos_w_np)
                .float()
                .to(self.device)
                .unsqueeze(0)
            )
            current_ee_quat_wxyz_w = (
                torch.from_numpy(current_ee_quat_wxyz_w_np)
                .float()
                .to(self.device)
                .unsqueeze(0)
            )

            # Check convergence
            pos_error = torch.linalg.norm(target_ee_pos_w - current_ee_pos_w)
            # Orientation error: angle of (q_target * q_current_inv)
            error_quat_wxyz = quat_mul(
                target_ee_quat_wxyz_w, quat_inv(current_ee_quat_wxyz_w)
            )
            # Angle from axis-angle representation: 2 * acos(w) or norm of axis-angle vector
            # Using axis_angle_from_quat which returns axis * angle
            axis_angle_error = axis_angle_from_quat(error_quat_wxyz)  # Shape (1,3)
            orient_error_rad = torch.linalg.norm(axis_angle_error, dim=1)  # Shape (1)

            if (
                pos_error < pos_threshold_abs
                and orient_error_rad < orient_threshold_abs
            ):
                logger.debug(
                    f"Arm movement converged in {step_idx} steps. Pos error: {pos_error.item():.4f}, Orient error: {orient_error_rad.item():.4f} rad"
                )
                break

            # Compute delta for DIK (relative command)
            # Delta position in world frame
            delta_pos_w = target_ee_pos_w - current_ee_pos_w

            # Delta orientation (axis-angle) in world frame for one step
            # We want to move towards target_ee_quat_wxyz_w from current_ee_quat_wxyz_w
            # The delta_rot for DIK is an axis-angle representing the desired *change* in orientation this step
            # Let's use the error_quat_wxyz and scale its rotation.
            # A small step towards the target orientation:
            # Scale down the axis_angle_error. Max rotation per step can be configured.
            max_rot_per_step_rad = self.env_cfg.get("max_rotation_per_step_rad", 0.1)
            # Clamp delta_rot magnitude
            current_rot_step_rad = torch.clamp(
                orient_error_rad, max=max_rot_per_step_rad
            )
            delta_rot_axis_angle_w = axis_angle_error * (
                current_rot_step_rad / (orient_error_rad + 1e-6)
            )  # Scale to desired step size

            # Clamp delta_pos magnitude per step
            max_pos_per_step_m = self.env_cfg.get("max_translation_per_step_m", 0.05)
            current_pos_step_m = torch.clamp(pos_error, max=max_pos_per_step_m)
            delta_pos_w_clamped = delta_pos_w * (
                current_pos_step_m / (pos_error + 1e-6)
            )

            # Set command for DIK (delta_pos_w, delta_rot_axis_angle_w)
            # DIK expects command as [dx, dy, dz, dax, day, daz]
            self.dik_controller.set_command(
                torch.cat([delta_pos_w_clamped, delta_rot_axis_angle_w], dim=1)
            )

            # Compute and apply joint positions
            # Jacobian is w.r.t. base frame of the robot, for the specified end-effector
            jacobian_tensor = self.robot_articulation.get_jacobian(
                body_name=ee_prim_name
            )  # This should return a Tensor

            current_joint_pos_rad_np = (
                self.robot_articulation.get_joint_positions()
            )  # All joints
            current_joint_pos_rad = (
                torch.from_numpy(current_joint_pos_rad_np)
                .float()
                .to(self.device)
                .unsqueeze(0)
            )

            # DIK compute needs ee_pos, ee_quat, jacobian, joint_pos
            # Ensure shapes match DIK expectations (likely batched [1, N])
            desired_joint_pos_rad = self.dik_controller.compute(
                ee_pos=current_ee_pos_w,
                ee_quat=current_ee_quat_wxyz_w,
                jacobian=jacobian_tensor.unsqueeze(0),  # Add batch dim if not present
                joint_pos=current_joint_pos_rad,  # Pass all joint positions
            )

            # Apply to arm joints only (DIK should only affect arm joints if configured right)
            # The UR5 set_joint_positions_without_gripper handles this internally if it knows arm joint indices
            self.robot_articulation.set_joint_positions_without_gripper(
                desired_joint_pos_rad.squeeze(0).cpu().numpy()
            )

            self.world.step(render=True)
            # time.sleep(dt) # If running faster than real-time, this might be needed, or adjust control_freq
        else:  # Loop finished without break
            logger.warning(
                f"Arm movement timed out after {max_steps} steps. Pos error: {pos_error.item():.4f}, Orient error: {orient_error_rad.item():.4f} rad"
            )

        # --- Gripper Actuation Phase ---
        if gripper_command != 0:  # If command is to open or close
            target_gripper_angles = (
                self.gripper_open_angles_rad
                if gripper_command > 0
                else self.gripper_closed_angles_rad
            )
            target_gripper_is_open = gripper_command > 0

            if (
                self.gripper_is_open != target_gripper_is_open
            ):  # Only actuate if state needs to change
                self.robot_articulation.set_gripper_positions(target_gripper_angles)
                self.gripper_is_open = target_gripper_is_open

                gripper_settle_steps = self.env_cfg.get("gripper_settle_steps", 10)
                for _ in range(gripper_settle_steps):
                    self.world.step(render=True)
                logger.debug(
                    f"Gripper action: {'open' if gripper_command > 0 else 'close'}. New state: {'open' if self.gripper_is_open else 'closed'}"
                )
            else:
                logger.debug(
                    f"Gripper already in desired state: {'open' if self.gripper_is_open else 'closed'}"
                )

        # --- Final Observation ---
        final_obs_dict = self._process_obs(do_vlm=True)
        self.latest_obs.update(final_obs_dict)

        # Rewards and termination are not yet implemented
        self.latest_reward = 0.0
        self.latest_terminate = False

        return self.latest_obs, self.latest_reward, self.latest_terminate

    def move_to_pose(
        self,
        pose_xyz_wxyz: Union[np.ndarray, List[float]],
    ) -> Tuple[Observation, Optional[float], Optional[bool]]:
        pose_np = np.asarray(pose_xyz_wxyz, dtype=np.float32)  # xyz_wxyz
        # Maintain current gripper state by using a "hold" command (0)
        action = np.concatenate([pose_np, [0.0]])  # gripper command 0.0 for hold
        return self.apply_action(action)

    def _trigger_gripper_action(
        self,
        gripper_command: float,  # -1 close, 1 open
    ) -> Tuple[Observation, Optional[float], Optional[bool]]:
        current_ee_state = self._get_robot_obs()  # Get fresh EE pose
        ee_pos = current_ee_state.get("ee_pos_w", self.initial_ee_pos)
        ee_quat_wxyz = current_ee_state.get("ee_quat_w", self.initial_ee_quat_wxyz)

        action = np.concatenate(
            [np.asarray(ee_pos), np.asarray(ee_quat_wxyz), [gripper_command]]
        )
        return self.apply_action(action)

    def open_gripper(self) -> Tuple[Observation, Optional[float], Optional[bool]]:
        return self._trigger_gripper_action(1.0)

    def close_gripper(self) -> Tuple[Observation, Optional[float], Optional[bool]]:
        return self._trigger_gripper_action(-1.0)

    def set_gripper_state(
        self,
        gripper_open_fraction: float,  # 0.0 closed, 1.0 open
    ) -> Tuple[Observation, Optional[float], Optional[bool]]:
        command = (
            1.0
            if gripper_open_fraction > self.env_cfg.get("gripper_open_threshold", 0.5)
            else -1.0
        )
        return self._trigger_gripper_action(command)

    def reset_to_default_pose(
        self,
    ) -> Tuple[Observation, Optional[float], Optional[bool]]:
        # Target the initial EE pose and current gripper state
        gripper_cmd = (
            1.0 if self.initial_gripper_is_open else -1.0
        )  # This should be what reset() establishes
        action = np.concatenate(
            [self.initial_ee_pos, self.initial_ee_quat_wxyz, [gripper_cmd]]
        )
        return self.apply_action(action)

    def get_ee_pose(self) -> Optional[np.ndarray]:  # Returns xyz_wxyz
        obs_data = self.latest_obs.get(
            "_data", {}
        )  # Access internal dict to avoid resolving tasks if not needed
        pos = obs_data.get("ee_pos_w")
        quat = obs_data.get("ee_quat_w")
        if pos is not None and quat is not None:
            return np.concatenate([np.asarray(pos), np.asarray(quat)])
        # Fallback to direct query if obs is not populated yet (e.g. before first reset)
        ee_prim_name = self.robot_cfg["robot"]["end_effector_prim_name"]
        pos_np, quat_wxyz_np = self.robot_articulation.get_end_effector_pose(
            body_name=ee_prim_name
        )
        return np.concatenate([pos_np, quat_wxyz_np])

    def get_object_names(self) -> List[str]:  # From EnvIsaacLab
        return list(self.env_cfg.get("scene_target_objects", []))

    def get_3d_obs_by_name_by_vlm(
        self, query_name: str
    ) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        # Largely same as EnvIsaacLab, ensure camera data is correctly accessed
        if not self.use_vlm:
            logger.warning("VLM is disabled; cannot get 3D obs by VLM.")
            return None
        if not self.latest_obs.get("camera"):
            logger.error("No camera data in latest_obs for VLM.")
            return None

        camera_data = self.latest_obs["camera"]
        vlm_masks_combined = camera_data.get("masks")
        if vlm_masks_combined is None:
            logger.warning("No VLM masks found in camera data.")
            return None
        if not isinstance(vlm_masks_combined, np.ndarray):
            logger.error(
                f"VLM masks are not a numpy array, type: {type(vlm_masks_combined)}"
            )
            return None

        all_points, all_masks_flat, all_normals = [], [], []
        for idx, cam_name in enumerate(
            self.cameras_to_use_names
        ):  # Use configured camera names
            points_w_hxwxd = camera_data.get(f"{cam_name}_pointcloud")  # HxWx3
            if points_w_hxwxd is None:
                continue
            points_w_flat = points_w_hxwxd.reshape(-1, 3)  # Nx3

            cam_mask_hw = vlm_masks_combined[idx]  # (H, W)
            if points_w_flat.shape[0] != cam_mask_hw.size:
                logger.error(
                    f"Point cloud size {points_w_flat.shape[0]} != mask size {cam_mask_hw.size} for {cam_name}"
                )
                continue

            all_points.append(points_w_flat)
            all_masks_flat.append(cam_mask_hw.reshape(-1))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_w_flat)
            if len(pcd.points) >= 3:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.05, max_nn=30
                    )
                )
                normals = np.asarray(pcd.normals)
                lookat_vec = self.camera_info.get(f"{cam_name}_lookat")
                if lookat_vec is not None:
                    dot_prod = np.sum(normals * np.asarray(lookat_vec), axis=1)
                    normals[dot_prod > 0] *= -1  # Flip normals to point towards camera
            else:
                normals = np.zeros_like(points_w_flat)
            all_normals.append(normals)

        if not all_points:
            return None
        combined_points = np.concatenate(all_points, axis=0)
        combined_masks_flat = np.concatenate(all_masks_flat, axis=0)
        combined_normals = np.concatenate(all_normals, axis=0)

        target_category_id = self.name2categerylabel.get(query_name)
        if target_category_id is None:
            logger.error(f"Query name {query_name} not in name2categorylabel map.")
            return None

        category_ids_from_mask = combined_masks_flat // self.category_multiplier
        instance_ids_from_mask = combined_masks_flat % self.category_multiplier
        category_match_mask = category_ids_from_mask == target_category_id
        unique_instances_in_category = np.unique(
            instance_ids_from_mask[category_match_mask]
        )
        unique_instances_in_category = unique_instances_in_category[
            unique_instances_in_category > 0
        ]

        found_objects_pcd_list: List[Tuple[np.ndarray, np.ndarray]] = []
        for inst_id in unique_instances_in_category:
            instance_specific_mask = category_match_mask & (
                instance_ids_from_mask == inst_id
            )
            obj_points = combined_points[instance_specific_mask]
            obj_normals = combined_normals[instance_specific_mask]
            if obj_points.shape[0] == 0:
                continue

            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(obj_points)
            pcd_obj.normals = o3d.utility.Vector3dVector(obj_normals)
            pcd_downsampled = pcd_obj.voxel_down_sample(
                voxel_size=self.env_cfg.get("vlm_voxel_size", 0.005)
            )
            pcd_filtered, _ = pcd_downsampled.remove_statistical_outlier(
                nb_neighbors=self.env_cfg.get("vlm_stat_neighbors", 20),
                std_ratio=self.env_cfg.get("vlm_stat_std_ratio", 2.0),
            )
            if len(pcd_filtered.points) > 0:
                found_objects_pcd_list.append(
                    (np.asarray(pcd_filtered.points), np.asarray(pcd_filtered.normals))
                )
        logger.info(
            f"Found {len(found_objects_pcd_list)} instances of '{query_name}' via VLM."
        )
        return found_objects_pcd_list

    def turn_off_vlm(self):  # From EnvIsaacLab
        self.use_vlm = False
        logger.info("VLM has been turned off.")

    def close(self):
        """Clean up resources, especially the simulation world."""
        if self.world:
            # If your RobotController has a cleanup method, call it.
            # self.robot_controller.cleanup()
            # Isaac Sim's World doesn't have a close(), but stopping physics & clearing stage might be relevant
            # if this env object manages the whole app lifecycle.
            # For now, assume lifecycle is managed externally.
            pass
        logger.info(
            "EnvIsaacSim closed (placeholder, actual cleanup might be in RobotController or app level)."
        )


if __name__ == "__main__":
    from omni.isaac.kit import SimulationApp
    # KIT_CONFIG = {"renderer": "RayTracedLighting", "headless": False}
    # simulation_app = SimulationApp(KIT_CONFIG) # If running standalone

    # This needs to be adjusted based on how your RobotController's ConfigManager works
    # and the actual paths to your USDs and config files.
    # The structure below is a guess based on your RobotController's __init__.

    # Example configuration for EnvIsaacSim
    # It's composed of robot_controller_cfg (for your RobotController)
    # and environment_cfg (for EnvIsaacSim specific settings)

    # Get path to this script to build relative paths for USDs
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Define paths to your assets ---
    # You NEED to replace these with actual paths to your UR5 USD and any scene USD
    # For example, if you have 'assets/ur5e/ur5e.usd' relative to this script:
    robot_usd_rel_path = "assets/ur5e_rg2_better_physics_temp/ur5e_rg2_instanceable.usd"  # EXAMPLE: Replace!
    robot_usd_abs_path = os.path.join(current_script_dir, robot_usd_rel_path)

    # scene_usd_rel_path = "assets/scene/my_scene.usd" # EXAMPLE: Replace or set to None
    # scene_usd_abs_path = os.path.join(current_script_dir, scene_usd_rel_path) if scene_usd_rel_path else None
    scene_usd_abs_path = (
        None  # No extra scene for this example, robot loads into empty world
    )

    if not os.path.exists(robot_usd_abs_path):
        print(
            f"ERROR: Robot USD not found at {robot_usd_abs_path}. Please set correct path."
        )
        # simulation_app.close() # if used
        exit()

    sim_env_cfg = {
        "robot_controller_cfg": {  # This structure should match what your RobotController expects
            # "config_file_path": "path/to/your_robot_controller_config.yaml", # If RobotController uses a YAML
            # OR, provide parameters directly if RobotController accepts a dict:
            "robot_usd_path": robot_usd_abs_path,
            "end_effector_prim_name": "tool0",  # Check this in your UR5 USD
            "initial_joint_angles_deg": [
                0,
                -70,
                110,
                -125,
                -90,
                0,
            ],  # UR5 specific: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
            "base_position": [0.0, 0.0, 0.0],  # Base position of the robot in the world
            "use_gripper": True,
            # Gripper joint names for UR5 with RG2 (example, verify with your USD)
            "gripper_joint_names": [
                "finger_joint",
                "right_outer_knuckle_joint",
            ],  # Simplified for some UR5+RG2 USDs. Check YOUR USD.
            # "left_inner_finger_joint", "right_inner_finger_joint",
            # "left_inner_knuckle_joint", "right_inner_knuckle_joint",
            # "left_outer_finger_joint", "right_outer_finger_joint", ],
            "gripper_open_angles_rad": [
                0.0,
                0.0,
            ],  # Example, radians. For RG2: usually one or two actuated joints.
            "gripper_closed_angles_rad": [0.8, 0.8],  # Example, radians.
            "arm_joint_names": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "scene_usd_path": scene_usd_abs_path,  # Path to a base scene if any
            "default_target_cube_pos": [
                0.5,
                0.0,
                0.2,
            ],  # For the FollowTarget task's target
        },
        "environment_cfg": {  # For EnvIsaacSim class itself
            "scene_target_objects": [
                "TargetCube"
            ],  # Name of the object created by FollowTarget task
            "cameras_to_use": ["front"],  # "wrist", "front"
            "vlm_service_base_url": "http://127.0.0.1:5000",
            "vlm_request_timeout": 10,
            "use_vlm": False,  # Set to True to test VLM
            "default_viewport_eye": [1.2, 1.2, 1.0],
            "default_viewport_target": [0.5, 0.0, 0.3],
            "movement_timeout_s": 7.0,
            "gripper_timeout_s": 3.0,  # Not directly used, gripper steps are fixed
            "position_threshold_abs": 0.02,  # For DIK convergence
            "orientation_threshold_abs_rad": 0.1,  # For DIK convergence
            "control_frequency_hz": 20.0,
            "gripper_settle_steps": 15,
            "gripper_open_threshold": 0.5,  # For set_gripper_state() mapping fraction to command
            "camera_configurations": {  # Detailed camera setups
                "wrist": {
                    "prim_path": "/World/UR5e/tool0/wrist_camera",  # Path for the camera prim
                    "position_offset": [
                        0.0,
                        0.0,
                        0.12,
                    ],  # Relative to EE link (tool0) Z-axis forward from tool0
                    "orientation_quat_wxyz_offset": [
                        0.7071,
                        0,
                        0.7071,
                        0,
                    ],  # Look along EE's Y axis (example: [w,x,y,z] for 90deg rot around Z)
                    # This is Gf.Quatd(0.7071, 0, 0.7071, 0) = 90 deg about Y of parent.
                    # To make camera look along X of parent: Gf.Quatd(0.7071, 0, -0.7071, 0)
                    # To make camera look along Z of parent: Gf.Quatd(0.7071, -0.7071, 0, 0)
                    # A common wrist cam looks "down" the Z axis of the tool.
                    # Isaac Sim default tool0 Z points "out". If cam looks along tool0 Z: identity quat.
                    # If optical frame is Z-fwd, Y-down, then from OpenGL (Z-out, Y-up) need 180deg about X.
                    # [0,1,0,0] is 180 deg about X. (w,x,y,z)
                    "resolution": [256, 256],
                    "attach_to_ee": True,
                },
                "front": {
                    "prim_path": "/World/front_camera",
                    "position": [0.8, 0.0, 0.9],
                    "orientation_quat_wxyz": [
                        0.653,
                        0.271,
                        0.653,
                        -0.271,
                    ],  # Look somewhat down and at origin
                    "resolution": [480, 640],
                    "attach_to_ee": False,
                },
            },
        },
    }

    try:
        logger.info(f"Initializing EnvIsaacSim...")
        env = EnvIsaacSim(cfg=sim_env_cfg)

        logger.info("Resetting environment...")
        obs = env.reset()
        current_ee_pose = env.get_ee_pose()
        logger.info(
            f"Initial EE Pose from obs: {obs.get('ee_pos_w')} {obs.get('ee_quat_w')}"
        )
        logger.info(
            f"Initial EE Pose from get_ee_pose: {current_ee_pose[:3]} {current_ee_pose[3:]}"
        )
        logger.info(f"Initial Gripper Open: {obs.get('gripper_open')}")

        if obs.get("camera"):
            front_cam_data = obs["camera"].get("front_rgb")
            if front_cam_data is not None:
                logger.info(f"Front camera RGB shape: {front_cam_data.shape}")
            if env.use_vlm and obs["camera"].get("masks") is not None:
                logger.info(f"Initial VLM Masks shape: {obs['camera']['masks'].shape}")

        # Example actions
        target_ee_xyz = np.array([0.4, 0.2, 0.3])
        target_ee_quat_wxyz = np.array(
            [0, 1.0, 0.0, 0.0]
        )  # Pointing down (180 deg about world X)

        logger.info(f"\nMoving to pose: {target_ee_xyz}, {target_ee_quat_wxyz}")
        obs, reward, terminated = env.move_to_pose(
            np.concatenate([target_ee_xyz, target_ee_quat_wxyz])
        )
        current_ee_pose = env.get_ee_pose()
        logger.info(
            f"Move result - EE Pose: {current_ee_pose[:3]} {current_ee_pose[3:]}, Gripper: {obs.get('gripper_open')}"
        )

        logger.info("\nClosing gripper...")
        obs, reward, terminated = env.close_gripper()
        logger.info(f"Close gripper result - Gripper Open: {obs.get('gripper_open')}")

        # Move to another pose
        target_ee_xyz_2 = np.array([0.5, -0.2, 0.4])
        target_ee_quat_wxyz_2 = np.array([0.7071, 0.7071, 0, 0])  # 90 deg about world X
        logger.info(
            f"\nMoving to pose 2: {target_ee_xyz_2}, {target_ee_quat_wxyz_2} while gripper closed"
        )
        obs, reward, terminated = env.move_to_pose(
            np.concatenate([target_ee_xyz_2, target_ee_quat_wxyz_2])
        )
        current_ee_pose = env.get_ee_pose()
        logger.info(
            f"Move result - EE Pose: {current_ee_pose[:3]} {current_ee_pose[3:]}, Gripper: {obs.get('gripper_open')}"
        )

        logger.info("\nOpening gripper...")
        obs, reward, terminated = env.open_gripper()
        logger.info(f"Open gripper result - Gripper Open: {obs.get('gripper_open')}")

        if (
            env.use_vlm and "TargetCube" in env.target_objects
        ):  # Assuming TargetCube is the default name
            logger.info("\nGetting 3D VLM observation for 'TargetCube'...")
            cube_3d_data = env.get_3d_obs_by_name_by_vlm("TargetCube")
            if cube_3d_data:
                for i, (points, normals) in enumerate(cube_3d_data):
                    logger.info(
                        f"  TargetCube Instance {i + 1}: {points.shape[0]} points, Normals shape: {normals.shape}"
                    )
            else:
                logger.info("  'TargetCube' not found by VLM or error occurred.")

        logger.info("\nResetting to default EE pose...")
        obs, reward, terminated = env.reset_to_default_pose()
        current_ee_pose = env.get_ee_pose()
        logger.info(
            f"Reset to default pose result - EE Pose: {current_ee_pose[:3]} {current_ee_pose[3:]}, Gripper: {obs.get('gripper_open')}"
        )

    except Exception as e_main:
        logger.error(f"Main execution error: {e_main}", exc_info=True)
    finally:
        # if simulation_app:
        #     simulation_app.close() # Ensure simulation app is closed cleanly
        # logger.info("SimulationApp closed.")
        # If your EnvIsaacSim or RobotController holds onto the world, you might want a env.close()
        if "env" in locals() and hasattr(env, "close"):
            env.close()

    logger.info("Example script for EnvIsaacSim finished.")
