# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import inspect
from typing import Any, Dict, Optional, List, Tuple, Union
import gymnasium as gym
import os
import torch
import logging
import base64
import requests  # requests is synchronous
import numpy as np
import open3d as o3d

# import logging # duplicate import
import time
import functools  # For the decorator

import carb

from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.app import get_app
import omni.kit.async_engine  # For a robust async_to_sync

from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.data_collector import RobomimicDataCollector
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

from omni.isaac.lab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import (
    FrankaCubeLiftEnvCfg,
)  # Assuming this specific Cfg, adjust if EnvIsaacLab is more general
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.sensors.camera import TiledCamera
from omni.isaac.lab.sensors.camera.utils import (
    save_images_to_file,
    create_pointcloud_from_depth,
    transform_points,
)
from omni.isaac.lab.assets.articulation.articulation import Articulation
from omni.isaac.lab.utils.math import (
    subtract_frame_transforms,
    quat_rotate,
    euler_xyz_from_quat,
    quat_from_matrix,
    quat_mul,
    quat_from_angle_axis,
    quat_inv,
    matrix_from_quat,
)

# import omni.replicator.core as rep # Not used in the provided snippet
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.xforms import reset_and_set_xform_ops
from pxr import Gf
from .mocap_manager import MocapManager
from .config_manager import ConfigManager


# Define the async_to_sync decorator for OmniKit
def omni_async_to_sync(f):
    """
    Decorator to run an async function synchronously within the OmniKit environment.
    It ensures that the function is run in OmniKit's asyncio loop and processes
    app updates while waiting for completion.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        coro = f(*args, **kwargs)
        task = omni.kit.async_engine.run_coroutine(coro)

        while not task.done():
            app_interface = get_app()
            if app_interface:
                app_interface.update()
            else:
                time.sleep(0.001)  # Minimal sleep if app is not available

        if task.exception():
            original_exception = task.exception()
            raise type(original_exception)(str(original_exception)).with_traceback(
                original_exception.__traceback__
            )
        return task.result()

    return wrapper


async_to_sync = omni_async_to_sync


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Use a module-level logger


class Observation:
    """可自动处理协程的观察数据类 (Observation data class that can automatically handle coroutines)"""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def update(self, other: Dict[str, Any]) -> None:
        """更新观察数据 (Update observation data)"""
        for k, v in other.items():
            if isinstance(v, dict):
                if k not in self._data or not isinstance(self._data[k], Observation):
                    self._data[k] = Observation()
                # Ensure self._data[k] is an Observation instance before calling update
                if isinstance(self._data[k], Observation):
                    self._data[k].update(v)
                else:  # Should not happen due to above check, but as a safeguard
                    new_obs = Observation()
                    new_obs.update(v)
                    self._data[k] = new_obs
            else:
                if inspect.iscoroutine(v):
                    task = omni.kit.async_engine.run_coroutine(v)
                    self._data[k] = task
                else:
                    self._data[k] = v

    def _resolve_value(self, value: Any) -> Any:
        """Helper to resolve asyncio.Task to its result, processing app updates."""
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
        """Internal getter that resolves tasks and wraps dicts."""
        if key not in self._data:
            raise KeyError(f"Key '{key}' not found in Observation data.")

        value = self._data[key]
        value = self._resolve_value(value)

        if isinstance(value, dict):  # If resolved value is a dict, wrap it
            obs_wrapper = Observation()
            obs_wrapper.update(value)
            self._data[key] = obs_wrapper  # Store wrapped version for future accesses
            return obs_wrapper

        # If original was a task and now resolved, update stored value
        if isinstance(self._data[key], asyncio.Task) and not isinstance(
            value, asyncio.Task
        ):
            self._data[key] = value

        return value

    def __getitem__(self, key):
        return self._getitem_internal(key)

    def __setitem__(self, key, value):
        """支持字典式赋值 (Support dictionary-style assignment)"""
        if isinstance(value, dict):
            obs = Observation()
            obs.update(value)
            self._data[key] = obs
        elif inspect.iscoroutine(value):
            task = omni.kit.async_engine.run_coroutine(value)
            self._data[key] = task
        else:
            self._data[key] = value

    def get(self, key, default=None):
        """获取值，支持默认值 (Get value, support default value)"""
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key in self._data

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Observation object to a Python dictionary, resolving any pending tasks."""
        resolved_dict = {}
        for (
            k_loop,
            v_initial_loop,
        ) in self.items():  # Use items() to trigger resolution via __getitem__
            if isinstance(v_initial_loop, Observation):
                resolved_dict[k_loop] = v_initial_loop.to_dict()
            else:
                resolved_dict[k_loop] = v_initial_loop
        return resolved_dict

    def items(self):
        # Resolve all values before yielding items
        for key_ in list(self._data.keys()):  # Iterate over a copy of keys
            yield key_, self[key_]  # self[key] will resolve tasks and wrap dicts


class EnvIsaacLab:
    def __init__(self, task_name: str, cfg: dict, visualizer=None):
        # parse_env_cfg usually handles finding registered configs.
        env_cfg: FrankaCubeLiftEnvCfg = parse_env_cfg(  # type: ignore
            task_name
        )

        self.cfg = cfg  # User-provided config for the EnvIsaacLab wrapper class

        # Modify env_cfg as per original logic
        env_cfg.terminations.time_out.time_out = False
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.episode_length_s = (
            1.0e9  # Effectively disable episode length termination
        )

        env_cfg.observations.policy.concatenate_terms = False
        env_cfg.observations.policy.enable_corruption = False

        env_cfg.observations.rgb.concatenate_terms = False

        env_cfg.commands.object_pose.debug_vis = False

        self.env = gym.make(task_name, cfg=env_cfg)
        self._set_camera_view_to_target_object()
        self.env.unwrapped.sim.step()  # type: ignore

        self.target_objects = self.get_object_names()
        self.cameras = self.cfg.get("cameras_to_use", ["wrist", "front"])
        self.camera_info: Dict[str, Any] = {}
        self.category_multiplier = self.cfg.get("vlm_category_multiplier", 100)
        self.name2categerylabel = {
            name: i for i, name in enumerate(self.target_objects, start=1)
        }
        self.categerylabel2name = {
            i: name for i, name in enumerate(self.target_objects, start=1)
        }

        self.device = self.env.unwrapped.action_manager.device  # type: ignore
        self.latest_obs = Observation()
        self.init_obs = (
            Observation()
        )  # Stores the observation from the very first reset

        self.workspace_bounds_min = np.array(
            self.cfg.get("workspace_bounds_min", [-0.1, -0.5, 0.0])
        )
        self.workspace_bounds_max = np.array(
            self.cfg.get("workspace_bounds_max", [1.0, 0.5, 1.0])
        )
        self.visualizer = visualizer
        if self.visualizer is not None:
            self.visualizer.update_bounds(
                self.workspace_bounds_min, self.workspace_bounds_max
            )
        self.use_vlm = self.cfg.get("use_vlm", True)
        self.latest_action: Optional[np.ndarray] = None
        self.latest_reward: Optional[float] = None
        self.latest_terminate: Optional[bool] = None

        # 初始化动作捕捉
        self.config_manager = ConfigManager(None)
        self.mocap_manager = MocapManager(
            self.config_manager,
            prim_prefix="",  # PN_Stickman_v12_ThumbInward"
        )
        if not self.mocap_manager.setup_scene(world=self.env.unwrapped):
            logger.error("设置动作捕捉场景失败")
            return False

        logger.info(
            f"EnvIsaacLab initialized. Target objects: {self.target_objects}, VLM: {self.use_vlm}"
        )

    def turn_off_vlm(self):
        self.use_vlm = False
        logger.info("VLM has been turned off.")

    def _set_camera_view_to_target_object(self):
        viewport = get_active_viewport()
        if viewport:
            set_camera_view(
                eye=self.cfg.get("default_viewport_eye", [1.6, 0.0, 0.6]),
                target=self.cfg.get("default_viewport_target", [0.0, 0.0, 0.0]),
                viewport_api=viewport,
            )
        else:
            logger.warning("No active viewport found to set camera view.")

    def _request_vlm_service(
        self, endpoint: str, labels: List[str], data_array: np.ndarray
    ) -> Optional[np.ndarray]:
        """Helper function to make requests to the VLM service."""
        url = f"{self.cfg.get('vlm_service_base_url', 'http://127.0.0.1:5000')}/{endpoint}"
        data_bytes = data_array.tobytes()
        data_base64 = base64.b64encode(data_bytes).decode("utf-8")
        payload = {"label": labels, "data": data_base64, "shape": data_array.shape}

        try:
            resp = requests.post(
                url, json=payload, timeout=self.cfg.get("vlm_request_timeout", 10)
            )
            resp.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            result = resp.json()
            return np.array(result).astype(np.uint32)
        except requests.exceptions.RequestException as e:
            logger.error(f"VLM request to {url} failed: {e}")
            return None
        except ValueError as e:  # Includes JSONDecodeError
            logger.error(f"Failed to decode VLM response from {url}: {e}")
            return None

    def _request_process_sole_frame(
        self, labels: List[str], data_array: np.ndarray
    ) -> Optional[np.ndarray]:
        return self._request_vlm_service("process_sole_frame", labels, data_array)

    def _request_process_first_frame(
        self, labels: List[str], data_array: np.ndarray
    ) -> Optional[np.ndarray]:
        return self._request_vlm_service("process_first_frame", labels, data_array)

    def _request_process_frame(self, cam_obs: Dict[str, Any]) -> Dict[str, Any]:
        rgbs_list = [
            cam_obs[f"{cam}_rgb"] for cam in self.cameras if f"{cam}_rgb" in cam_obs
        ]
        if not rgbs_list:
            logger.warning("No RGB data found in cam_obs for VLM processing.")
            cam_obs["masks"] = None
            return cam_obs

        rgbs = np.stack(rgbs_list, axis=0)
        masks = self._request_vlm_service(
            "process_frame", self.target_objects, rgbs
        )  # Pass target_objects as default labels
        cam_obs["masks"] = masks
        return cam_obs

    def reset(self) -> Observation:
        logger.info("Resetting environment...")
        obs_dict_raw: Dict[str, Any] = {}
        obs_from_env, _ = self.env.reset()

        obs_dict_raw.update(self._extract_policy_obs(obs_from_env))

        raw_camera_data = self.get_camera_observation(self.env.unwrapped)  # type: ignore
        if self.use_vlm:
            rgbs_list = [
                raw_camera_data[f"{cam}_rgb"]
                for cam in self.cameras
                if f"{cam}_rgb" in raw_camera_data
            ]
            if rgbs_list:
                rgbs_stack = np.stack(rgbs_list, axis=0)
                masks = self._request_process_first_frame(
                    labels=self.target_objects, data_array=rgbs_stack
                )
                raw_camera_data["masks"] = masks
            else:
                logger.warning("No RGB data found for initial VLM frame processing.")
                raw_camera_data["masks"] = None

        obs_dict_raw["camera"] = raw_camera_data

        self.init_obs.update(obs_dict_raw)  # Store initial state
        self.latest_obs.update(obs_dict_raw)  # Current state is also initial state

        if self.use_vlm and (
            not raw_camera_data.get("masks") is not None
        ):  # Check if masks is None or empty
            logger.warning("VLM failed to process first frame or no masks returned.")
            # Depending on strictness, could raise error: raise ValueError("VLM failed on first frame.")

        self.latest_action = None
        self.latest_reward = None
        self.latest_terminate = False
        logger.info("Environment reset complete.")
        return self.latest_obs

    def get_object_names(self) -> List[str]:
        return list(self.cfg.get("scene_target_objects", []))

    def get_frame_transforms(
        self,
        frame_name: str,
        robot: Articulation,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        relative_translation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_indices = [
            idx for idx, name in enumerate(robot.data.body_names) if name == frame_name
        ]
        if not frame_indices:
            raise ValueError(
                f"Frame name '{frame_name}' not found in robot: {robot.data.body_names}"
            )
        frame_idx = frame_indices[0]

        # Ensure data is on CPU for consistency if further ops are numpy/cpu based
        frame_pos_w = robot.data.body_pos_w[:, frame_idx, :].cpu()
        frame_quat_w = robot.data.body_quat_w[:, frame_idx, :].cpu()

        # Ensure translation and rotation are tensors and on the same device as frame_pos_w
        translation_t = torch.as_tensor(
            translation, dtype=frame_pos_w.dtype, device=frame_pos_w.device
        )
        rotation_t = torch.as_tensor(
            rotation, dtype=frame_quat_w.dtype, device=frame_quat_w.device
        )

        # Handle batching: if frame_pos is (1,3) and translation is (3), unsqueeze translation
        if translation_t.dim() == 1 and frame_pos_w.dim() == 2:  # (3) vs (1,3)
            translation_t = translation_t.unsqueeze(0)
        if rotation_t.dim() == 1 and frame_quat_w.dim() == 2:  # (4) vs (1,4)
            rotation_t = rotation_t.unsqueeze(0)

        # Target orientation in world frame
        # quat_mul(A, B) applies B in A's frame. Here, apply local rotation to world orientation.
        target_quat_w = quat_mul(frame_quat_w, rotation_t)

        if relative_translation:
            # Translate along axes of the *target* orientation in world frame
            offset_w = quat_rotate(target_quat_w, translation_t)
            target_pos_w = frame_pos_w + offset_w
        else:
            # Translate in world frame axes
            target_pos_w = frame_pos_w + translation_t

        return target_pos_w, target_quat_w

    def _build_extrinsic_matrix(
        self, position: torch.Tensor, orientation: torch.Tensor
    ) -> torch.Tensor:
        pos = position.squeeze()
        orient = orientation.squeeze()
        if pos.shape != (3,) or orient.shape != (4,):
            raise ValueError(
                f"Position {pos.shape} or orientation {orient.shape} have incorrect dimensions."
            )

        R = matrix_from_quat(orient.unsqueeze(0)).squeeze(
            0
        )  # matrix_from_quat expects batch
        extrinsic_matrix = torch.eye(4, dtype=R.dtype, device=R.device)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = pos
        return extrinsic_matrix

    @async_to_sync  # Makes this method blocking
    async def get_camera_observation(
        self, env_unwrapped: ManagerBasedEnv
    ) -> Dict[str, Any]:
        logger.debug("Starting camera observation (async core, sync wrapper)")

        q_x_180 = quat_from_angle_axis(
            torch.tensor(torch.pi), torch.tensor([1.0, 0, 0], dtype=torch.float)
        )
        q_z_n90 = quat_from_angle_axis(
            torch.tensor(-0.5 * torch.pi), torch.tensor([0, 0, 1], dtype=torch.float)
        )
        q_z_90 = quat_from_angle_axis(
            torch.tensor(0.5 * torch.pi), torch.tensor([0, 0, 1], dtype=torch.float)
        )
        q_y_60 = quat_from_angle_axis(
            torch.tensor(torch.pi / 3.0), torch.tensor([0, 1, 0], dtype=torch.float)
        )

        robot_articulation = env_unwrapped.scene.articulations.get("robot")
        if not robot_articulation:
            logger.error("Robot articulation not found for camera observation.")
            return {}

        cam_pos_offset = torch.tensor((0.62562, -1.29608, 2.95021), dtype=torch.float)

        # Camera pose configurations, relative to robot parts or world
        cam_configs = {
            "wrist": {
                "parent_frame": "panda_hand",
                "robot": robot_articulation,
                "rotation": quat_mul(q_x_180, q_z_n90),  # Local rotation
                "translation": torch.tensor(
                    [0.0, 0.05, -0.06], dtype=torch.float
                ),  # Local translation
                "relative_translation": True,
            },
            "front": {  # Example: fixed world pose, but defined relative to link0 for convenience if robot moves significantly
                "parent_frame": "panda_link0",
                "robot": robot_articulation,
                "rotation": quat_mul(
                    q_y_60, q_z_90
                ),  # Rotation relative to link0's orientation
                "translation": torch.tensor(
                    [0, 0.0 - 1.5, 1.0], dtype=torch.float
                ),  # Offset from link0 origin in link0's frame if relative_translation=True
                # Or world offset if relative_translation=False
                "relative_translation": False,  # If this means translation is world offset from link0_pos
                # Or True if translation is in link0_frame. This needs clarification.
                # Assuming relative_translation=False implies translation is a world-offset ADDED to frame_pos
                # For a fixed world camera: parent_frame=None or a fixed world prim.
                # Let's assume 'front' is fixed relative to robot base for now.
            },
        }

        # For a truly fixed world camera, it's simpler:
        # "front_fixed": {"pos": torch.tensor([1.6,0,0.6]), "quat": quat_mul(q_y_60, q_z_90)}

        camera_sensor: Optional[TiledCamera] = env_unwrapped.scene.sensors.get(
            "tiled_camera"
        )  # type: ignore
        if not camera_sensor:
            logger.error("TiledCamera sensor not found.")
            return {}

        camera_data_dict: Dict[str, Any] = {}
        forward_vector_optical = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float
        )  # Z-forward in optical frame

        for cam_name, cfg in cam_configs.items():
            cam_pos_w, cam_quat_w = self.get_frame_transforms(
                cfg["parent_frame"],
                cfg["robot"],
                cfg["rotation"],
                cfg["translation"],
                cfg["relative_translation"],
            )

            camera_sensor.set_world_poses(
                positions=cam_pos_w, orientations=cam_quat_w, convention="opengl"
            )

            await env_unwrapped.sim.render_async()
            await get_app().next_update_async()
            camera_sensor.update(dt=0.0, force_recompute=True)

            intrinsic_m = camera_sensor.data.intrinsic_matrices.clone().cpu().squeeze(0)
            extrinsic_m = self._build_extrinsic_matrix(
                cam_pos_w.cpu(), cam_quat_w.cpu()
            )  # Already on CPU

            # lookat vector in world frame: camera's view direction in world
            # OpenGL camera looks along -Z. R_cw = extrinsic_m[:3,:3]. R_wc = R_cw.T
            # lookat_w = R_wc @ [0,0,-1]_cam_body
            lookat_w = extrinsic_m[:3, :3].T @ torch.tensor(
                [0.0, 0.0, -1.0], device=extrinsic_m.device, dtype=extrinsic_m.dtype
            )

            self.camera_info[f"{cam_name}_extrinsic_matrice"] = extrinsic_m.numpy()
            self.camera_info[f"{cam_name}_intrinsic_matrices"] = intrinsic_m.numpy()
            self.camera_info[f"{cam_name}_lookat"] = lookat_w.numpy()

            rgb = camera_sensor.data.output["rgb"].clone().cpu().squeeze(0)  # HxWxC
            depth = (
                camera_sensor.data.output["distance_to_image_plane"]
                .clone()
                .cpu()
                .squeeze(0)
            )  # HxWx1

            save_images_to_file(
                torch.tensor(rgb).unsqueeze(0) / 255.0,
                f"camera_rgb_{cam_name}.png",
            )
            save_images_to_file(
                torch.tensor(
                    (depth.clone() - depth.min()) / (depth.max() - depth.min())
                ).unsqueeze(0),
                f"camera_depth_{cam_name}.png",
            )
            # Pointcloud from depth: q_x_180 rotates from optical (Z fwd, Y down) to camera body (Z back, Y up)
            # This assumes depth values and intrinsics correspond to such an optical frame.
            pointcloud_cam_body = create_pointcloud_from_depth(
                intrinsic_matrix=intrinsic_m,
                depth=depth.squeeze(-1),
                orientation=q_x_180,
                keep_invalid=True,
            )
            pointcloud_w = transform_points(
                pointcloud_cam_body,
                orientation=cam_quat_w.squeeze().tolist(),
                position=cam_pos_w.squeeze().tolist(),
            )

            camera_data_dict[f"{cam_name}_rgb"] = rgb.permute(2, 0, 1).numpy()  # CxHxW
            camera_data_dict[f"{cam_name}_depth"] = depth.numpy()
            camera_data_dict[f"{cam_name}_pointcloud"] = (
                pointcloud_w.numpy()
                if isinstance(pointcloud_w, torch.Tensor)
                else pointcloud_w
            )

        logger.debug("Camera observation finished.")
        return camera_data_dict

    def get_3d_obs_by_name_by_vlm(
        self, query_name: str
    ) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        if not self.use_vlm:
            logger.warning("VLM is disabled; cannot get 3D obs by VLM.")
            return None
        if not self.latest_obs.get("camera"):
            logger.error("No camera data in latest_obs for VLM.")
            return None

        camera_data = self.latest_obs["camera"]  # This is an Observation object
        vlm_masks_combined = camera_data.get("masks")  # (num_cams, H, W)
        if vlm_masks_combined is None:
            logger.warning("No VLM masks found in camera data.")
            return None
        if not isinstance(
            vlm_masks_combined, np.ndarray
        ):  # Should be ndarray from VLM service
            logger.error(
                f"VLM masks are not a numpy array, type: {type(vlm_masks_combined)}"
            )
            return None

        all_points, all_masks_flat, all_normals = [], [], []
        for idx, cam_name in enumerate(self.cameras):
            points_w = camera_data.get(f"{cam_name}_pointcloud")  # (N, 3)
            if points_w is None:
                continue

            cam_mask_hw = vlm_masks_combined[idx]  # (H, W)
            if points_w.shape[0] != cam_mask_hw.size:  # Check total number of points
                logger.error(
                    f"Point cloud size {points_w.shape[0]} != mask size {cam_mask_hw.size} for {cam_name}"
                )
                continue

            all_points.append(points_w.reshape(-1, 3))
            all_masks_flat.append(cam_mask_hw.reshape(-1))

            # Normals estimation (can be slow, consider if pre-computed or needed)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_w.reshape(-1, 3))
            if len(pcd.points) >= 3:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.05, max_nn=30
                    )
                )
                normals = np.asarray(pcd.normals)
                # Flip normals based on camera lookat vector
                lookat_vec = self.camera_info.get(f"{cam_name}_lookat")
                if lookat_vec is not None:
                    dot_prod = np.sum(normals * np.asarray(lookat_vec), axis=1)
                    normals[dot_prod > 0] *= -1
            else:  # Not enough points for normal estimation
                normals = np.zeros_like(points_w.reshape(-1, 3))
            all_normals.append(normals)

        if not all_points:
            return None

        combined_points = np.concatenate(all_points, axis=0)
        combined_masks_flat = np.concatenate(all_masks_flat, axis=0)
        combined_normals = np.concatenate(all_normals, axis=0)

        # Filter by query_name
        target_category_id = self.name2categerylabel.get(query_name)
        if target_category_id is None:
            logger.error(f"Query name {query_name} not in name2categorylabel map.")
            return None

        # Extract category and instance IDs from masks
        category_ids_from_mask = combined_masks_flat // self.category_multiplier
        instance_ids_from_mask = combined_masks_flat % self.category_multiplier

        # Boolean mask for points belonging to the target category
        category_match_mask = category_ids_from_mask == target_category_id

        unique_instances_in_category = np.unique(
            instance_ids_from_mask[category_match_mask]
        )
        # Filter out instance ID 0 if it means "no instance" or "background"
        unique_instances_in_category = unique_instances_in_category[
            unique_instances_in_category > 0
        ]

        found_objects_pcd_list: List[Tuple[np.ndarray, np.ndarray]] = []
        for inst_id in unique_instances_in_category:
            # Boolean mask for this specific instance of the target category
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
                voxel_size=self.cfg.get("vlm_voxel_size", 0.005)
            )
            pcd_filtered, _ = pcd_downsampled.remove_statistical_outlier(
                nb_neighbors=self.cfg.get("vlm_stat_neighbors", 20),
                std_ratio=self.cfg.get("vlm_stat_std_ratio", 2.0),
            )
            if len(pcd_filtered.points) > 0:
                found_objects_pcd_list.append(
                    (np.asarray(pcd_filtered.points), np.asarray(pcd_filtered.normals))
                )

        logger.info(
            f"Found {len(found_objects_pcd_list)} instances of '{query_name}' via VLM."
        )
        return found_objects_pcd_list

    def get_scene_3d_obs(
        self, ignore_robot=False, ignore_grasped_obj=False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Note: ignore_robot and ignore_grasped_obj are not implemented here.
        # This would require segmentation information (e.g., from VLM or replicator).
        if not self.latest_obs.get("camera"):
            return np.array([]).reshape(0, 3), None

        points_list = []
        camera_data = self.latest_obs["camera"]
        for cam_name in self.cameras:
            cam_pc = camera_data.get(f"{cam_name}_pointcloud")
            if cam_pc is not None:
                points_list.append(cam_pc.reshape(-1, 3))

        return (
            np.concatenate(points_list, axis=0)
            if points_list
            else np.array([]).reshape(0, 3)
        ), None

    def _extract_policy_obs(self, obs_from_env: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to extract robot state (policy observations) from raw environment observation."""
        policy_data_dict: Dict[str, Any] = {}
        policy_obs = obs_from_env.get("policy")
        if isinstance(policy_obs, dict):
            if "ee_pos" in policy_obs and policy_obs["ee_pos"] is not None:
                policy_data_dict["ee_pos_w"] = policy_obs["ee_pos"][0][0].cpu().numpy()
            if "ee_quat" in policy_obs and policy_obs["ee_quat"] is not None:
                policy_data_dict["ee_quat_w"] = (
                    policy_obs["ee_quat"][0][0].cpu().numpy()
                )  # Should be wxyz

            if (
                "joint_pos_norm" in policy_obs
                and policy_obs["joint_pos_norm"] is not None
            ):
                joint_pos_norm = policy_obs["joint_pos_norm"][0]
                # Assuming 'robot' and 'panda_hand' are standard names. Make configurable if needed.
                robot_entity = self.env.unwrapped.scene["robot"]
                if robot_entity and "panda_hand" in robot_entity.actuators:
                    ee_joint_indices = robot_entity.actuators[
                        "panda_hand"
                    ].joint_indices
                    ee_joint_pos = joint_pos_norm[ee_joint_indices]
                    policy_data_dict["gripper_open"] = (
                        1.0 if torch.all(ee_joint_pos > 0.9) else 0.0
                    )
                else:
                    logger.warning(
                        "Panda hand actuator not found, cannot determine gripper state from joints."
                    )
                    policy_data_dict["gripper_open"] = (
                        0.0  # Default or last known state
                    )
        else:
            logger.warning(
                "Policy data missing or not a dict in observation from environment."
            )
        return policy_data_dict

    def apply_action(
        self, action_values: Union[np.ndarray, List[float]], relative_mode: bool = False
    ) -> Tuple[Observation, Optional[float], Optional[bool]]:
        action_np = np.asarray(
            action_values, dtype=np.float32
        )  # (target_ee_pose_xyz_wxyz, gripper_command)

        # Current EE pose for delta calculation or logging
        current_ee_pose_w = self.get_ee_pose()  # xyz_wxyz, can be None
        target_ee_pos_w = action_np[:3]

        action_tensor_for_env = self._process_action(
            action_np
        )  # batched tensor for env.step

        # Logic for deferring gripper closure
        gripper_command = action_np[-1]
        defer_gripper_closure = False
        if not relative_mode and gripper_command < 0.0:  # Close command
            current_gripper_status = self.latest_obs.get(
                "gripper_open", 0.0
            )  # Default to closed
            if current_gripper_status == 1.0:  # If gripper is open
                defer_gripper_closure = True
                # Create a temporary action tensor with neutral gripper command for arm movement phase
                temp_action_tensor = action_tensor_for_env.clone()
                temp_action_tensor[0, -1] = (
                    0.0  # Assuming 0.0 is neutral/hold for gripper
                )
                active_action_tensor = temp_action_tensor
                logger.info("Deferred gripper closure: moving arm first.")
            else:  # Gripper already closed or closing
                active_action_tensor = action_tensor_for_env
        else:  # Not a close command or relative mode (gripper handled directly)
            active_action_tensor = action_tensor_for_env

        # Arm movement loop
        start_time = time.time()
        movement_timeout = self.cfg.get("movement_timeout_s", 5.0)
        pos_threshold = self.cfg.get(
            "position_threshold", 0.02
        )  # For absolute mode convergence

        logger.debug(
            f"Applying action. Target EE pos: {target_ee_pos_w}. Relative: {relative_mode}."
        )

        last_obs_raw = None  # Store last raw observation from env.step
        while True:
            obs_raw, reward, terminated, truncated, info = self.env.step(
                active_action_tensor
            )
            last_obs_raw = obs_raw  # Keep track of the latest raw obs

            # Fast update of robot state during movement, no VLM/full camera processing yet
            policy_update = self._extract_policy_obs(obs_raw)
            self.latest_obs.update(policy_update)
            self.latest_reward = float(reward) if reward is not None else None
            self.latest_terminate = bool(
                terminated or truncated
            )  # Combine episode end conditions
            self.latest_action = active_action_tensor.squeeze(0).cpu().numpy()

            if self.latest_terminate:
                logger.info("Episode terminated during arm movement.")
                break

            # Check completion criteria
            if relative_mode:
                if (time.time() - start_time) > (
                    1.0 / self.cfg.get("relative_mode_control_freq_hz", 3.0)
                ):
                    break
            else:  # Absolute mode
                current_ee_pos = self.latest_obs.get("ee_pos_w")
                if current_ee_pos is not None:
                    error_norm = np.linalg.norm(
                        np.asarray(current_ee_pos) - target_ee_pos_w
                    )
                    if error_norm < pos_threshold:
                        logger.debug(
                            f"Absolute move target reached. Error: {error_norm:.4f}m"
                        )
                        break
                else:  # Should not happen if _extract_policy_obs works
                    logger.warning(
                        "Current EE position unavailable to check absolute move convergence."
                    )
                    break

            if (time.time() - start_time) > movement_timeout:
                logger.warning(f"Arm movement timed out after {movement_timeout}s.")
                break

        # Deferred gripper closure
        if defer_gripper_closure and not self.latest_terminate:
            logger.info("Executing deferred gripper closure.")
            # Use current EE pose for gripper action
            final_ee_pos = self.latest_obs.get(
                "ee_pos_w", target_ee_pos_w
            )  # Fallback to target if missing
            final_ee_quat = self.latest_obs.get(
                "ee_quat_w", action_np[3:7] if len(action_np) >= 7 else [1, 0, 0, 0]
            )

            close_action_np = np.concatenate(
                [np.asarray(final_ee_pos), np.asarray(final_ee_quat), [-1.0]]
            )
            close_action_tensor = self._process_action(close_action_np)

            gripper_timeout = self.cfg.get("gripper_timeout_s", 2.0)
            gripper_start_time = time.time()

            while (
                self.latest_obs.get("gripper_open", 1.0) == 1.0
            ):  # While gripper is open
                if (time.time() - gripper_start_time) > gripper_timeout:
                    logger.warning("Gripper closure timed out.")
                    break
                if (
                    self.latest_terminate
                ):  # Check again if episode terminated during previous step
                    logger.info("Episode terminated during gripper closure.")
                    break

                obs_raw, reward, terminated, truncated, info = self.env.step(
                    close_action_tensor
                )
                last_obs_raw = obs_raw

                policy_update = self._extract_policy_obs(obs_raw)
                self.latest_obs.update(policy_update)
                self.latest_reward = float(reward) if reward is not None else None
                self.latest_terminate = bool(terminated or truncated)
                self.latest_action = close_action_tensor.squeeze(0).cpu().numpy()

        # Final full observation update (with camera/VLM)
        if last_obs_raw is not None:  # If any step was taken
            logger.debug("Performing final full observation update post-action.")
            full_final_obs_data = self._process_obs(last_obs_raw, do_vlm=True)
            self.latest_obs.update(full_final_obs_data)
        else:  # No step was taken (e.g., episode terminated before movement)
            logger.warning(
                "No simulation step taken in apply_action, full observation may be stale."
            )

        return self.latest_obs, self.latest_reward, self.latest_terminate

    def move_to_pose(
        self,
        pose_xyz_wxyz: Union[np.ndarray, List[float]],
        speed: Optional[float] = None,
    ) -> Tuple[Observation, Optional[float], Optional[bool]]:
        pose_np = np.asarray(pose_xyz_wxyz, dtype=np.float32)  # xyz_wxyz
        # Maintain current gripper state
        last_gripper_cmd = (
            self.get_last_gripper_action()
        )  # Gets 1.0 (open) or -1.0 (closed)
        action = np.concatenate([pose_np, [last_gripper_cmd]])
        return self.apply_action(action, relative_mode=False)

    def _trigger_gripper_action(
        self, gripper_command: float
    ) -> Tuple[Observation, Optional[float], Optional[bool]]:
        ee_pos = self.latest_obs.get("ee_pos_w")
        ee_quat = self.latest_obs.get("ee_quat_w")  # Should be wxyz
        if ee_pos is None or ee_quat is None:
            logger.error(
                "Cannot trigger gripper: current EE pose unknown. Using initial pose as fallback."
            )
            ee_pos = self.init_obs.get(
                "ee_pos_w", [0.5, 0, 0.5]
            )  # Provide a safe default if init_obs also missing
            ee_quat = self.init_obs.get("ee_quat_w", [1, 0, 0, 0])  # wxyz

        action = np.concatenate(
            [np.asarray(ee_pos), np.asarray(ee_quat), [gripper_command]]
        )
        return self.apply_action(action, relative_mode=False)

    def open_gripper(self) -> Tuple[Observation, Optional[float], Optional[bool]]:
        return self._trigger_gripper_action(1.0)  # 1.0 for open

    def close_gripper(self) -> Tuple[Observation, Optional[float], Optional[bool]]:
        return self._trigger_gripper_action(-1.0)  # -1.0 for close

    def set_gripper_state(
        self, gripper_open_fraction: float
    ) -> Tuple[Observation, Optional[float], Optional[bool]]:
        # Maps a fraction (0.0 closed, 1.0 open) to command (-1.0 closed, 1.0 open)
        command = (
            1.0
            if gripper_open_fraction > self.cfg.get("gripper_open_threshold", 0.5)
            else -1.0
        )
        return self._trigger_gripper_action(command)

    def reset_to_default_pose(
        self,
    ) -> Tuple[Observation, Optional[float], Optional[bool]]:
        init_ee_pos = self.init_obs.get("ee_pos_w")
        init_ee_quat = self.init_obs.get("ee_quat_w")  # wxyz
        if init_ee_pos is None or init_ee_quat is None:
            logger.error("Initial EE pose not available for reset_to_default_pose.")
            return (
                self.latest_obs,
                self.latest_reward,
                self.latest_terminate,
            )  # Return current state

        pose_np = np.concatenate([np.asarray(init_ee_pos), np.asarray(init_ee_quat)])
        last_gripper_cmd = self.get_last_gripper_action()
        action = np.concatenate([pose_np, [last_gripper_cmd]])
        return self.apply_action(action, relative_mode=False)

    def get_ee_pose(self) -> Optional[np.ndarray]:  # Returns xyz_wxyz
        pos = self.latest_obs.get("ee_pos_w")
        quat = self.latest_obs.get("ee_quat_w")  # Expect wxyz
        if pos is not None and quat is not None:
            return np.concatenate([np.asarray(pos), np.asarray(quat)])
        return None

    def get_ee_pos(self) -> Optional[np.ndarray]:
        pose = self.get_ee_pose()
        return np.asarray(pose[:3]) if pose is not None else None

    def get_ee_quat(self) -> Optional[np.ndarray]:  # Returns wxyz
        pose = self.get_ee_pose()
        return np.asarray(pose[3:]) if pose is not None else None

    def get_last_gripper_action(self) -> float:  # Returns 1.0 (open) or -1.0 (closed)
        if self.latest_action is not None:
            return self.latest_action[-1]
        # Fallback to initial gripper state if no action taken yet
        init_gripper_open_status = self.init_obs.get(
            "gripper_open", 1.0
        )  # Default 1.0 (open)
        return 1.0 if init_gripper_open_status == 1.0 else -1.0

    def _process_obs(
        self, obs_from_env: Dict[str, Any], do_vlm: bool = True
    ) -> Dict[str, Any]:
        """Processes raw sim obs, adds camera data, and optionally VLM masks. Returns a dict."""
        processed_data_dict: Dict[str, Any] = {}
        processed_data_dict.update(self._extract_policy_obs(obs_from_env))

        raw_camera_data = self.get_camera_observation(self.env.unwrapped)  # type: ignore

        if self.use_vlm and do_vlm:
            # _request_process_frame expects a dict with RGB data and adds "masks" to it.
            # Ensure raw_camera_data is a mutable dict for _request_process_frame.
            cam_data_for_vlm = (
                raw_camera_data.copy() if isinstance(raw_camera_data, dict) else {}
            )
            processed_camera_data_with_vlm = self._request_process_frame(
                cam_data_for_vlm
            )
            processed_data_dict["camera"] = processed_camera_data_with_vlm
        else:
            # If not using VLM, ensure "masks" key exists and is None if expected by downstream.
            if isinstance(raw_camera_data, dict) and "masks" not in raw_camera_data:
                raw_camera_data["masks"] = None
            processed_data_dict["camera"] = raw_camera_data

        return processed_data_dict

    def _process_action(self, action_np: np.ndarray) -> torch.Tensor:
        # action_np: (target_ee_pos_xyz, target_ee_quat_wxyz, gripper_command)
        # gripper_command: 1.0 for open, -1.0 for close.
        action_tensor = torch.tensor(action_np.astype(np.float32), device=self.device)
        return action_tensor.unsqueeze(0)  # Add batch dimension

    def stop_preview(self):
        self.stop_preview = True

    def preview(self, max_step=100):
        self.stop_preview = False
        step = 0
        try:
            self.mocap_manager.move_model_to_origin()
            while not self.stop_preview:
                # Main loop for the simulation
                app_interface = get_app()
                app_interface.update()
                self.mocap_manager.update_avatar_posture()
                time.sleep(0.01)  # Sleep to avoid busy-waiting
                step += 1
                if step >= max_step:
                    logger.info("Max steps reached, stopping preview.")
                    break
        except KeyboardInterrupt:
            logger.info("Simulation loop interrupted by user.")
            self.env.close()


if __name__ == "__main__":
    # This block is for illustrative purposes.
    # To run this, Isaac Sim must be launched, and this script executed within its Python environment.
    # Example: ./python.sh my_env_isaac_lab_script.py --task FrankaCubeLift (if task arg is used)

    # Setup SimulationApp (essential for running Isaac Sim headless or with UI)
    from omni.isaac.kit import SimulationApp
    # KIT_CONFIG = {"renderer": "RayTracedLighting", "headless": False} # Example config
    # simulation_app = SimulationApp(KIT_CONFIG)

    # Default configuration for the EnvIsaacLab wrapper
    env_wrapper_cfg = {
        "scene_target_objects": [
            "Cube"
        ],  # Make sure 'Cube' matches object name in the scene
        "cameras_to_use": ["wrist", "front"],
        "vlm_service_base_url": "http://127.0.0.1:5000",  # VLM service URL
        "vlm_request_timeout": 10,  # Timeout for VLM requests
        "use_vlm": True,  # Enable or disable VLM usage
        "vlm_category_multiplier": 100,
        "vlm_voxel_size": 0.005,
        "vlm_stat_neighbors": 20,
        "vlm_stat_std_ratio": 1.0,
        "default_viewport_eye": [1.2, 1.2, 0.8],  # Camera position for the UI viewport
        "default_viewport_target": [
            0.5,
            0.0,
            0.3,
        ],  # What the UI viewport camera looks at
        "movement_timeout_s": 7.0,  # Max time for an arm movement action
        "gripper_timeout_s": 3.0,  # Max time for a gripper action
        "position_threshold": 0.02,  # Convergence threshold for absolute EE position
        "relative_mode_control_freq_hz": 5.0,  # Control frequency for relative mode actions
        "gripper_open_threshold": 0.5,  # Threshold to map continuous value to binary open/close
    }

    # Task name should match a registered Isaac Lab Task or be a path to its config YAML
    task_identifier = "FrankaCubeLift"  # Example: use the built-in lift task

    try:
        logger.info(f"Initializing EnvIsaacLab for task: {task_identifier}")
        env = EnvIsaacLab(task_name=task_identifier, cfg=env_wrapper_cfg)

        logger.info("Resetting environment...")
        obs = env.reset()
        logger.info(f"Initial EE Pose: {obs.get('ee_pos_w')} {obs.get('ee_quat_w')}")
        logger.info(f"Initial Gripper Open: {obs.get('gripper_open')}")
        if obs.get("camera") and obs["camera"].get("masks") is not None:
            logger.info(f"Initial VLM Masks shape: {obs['camera']['masks'].shape}")
        else:
            logger.info("No VLM masks in initial observation.")

        # Example actions
        target_ee_xyz = np.array([0.6, 0.1, 0.4])
        # IMPORTANT: Isaac Lab uses wxyz convention for quaternions in actions by default for franka ik controllers
        target_ee_quat_wxyz = np.array(
            [0.0, 1.0, 0.0, 0.0]
        )  # Rotate 180 deg around x-axis (example)
        # To get this from Euler (e.g. ZYX): q = q_z * q_y * q_x. Then ensure it's [w,x,y,z]

        full_pose_action = np.concatenate([target_ee_xyz, target_ee_quat_wxyz])

        logger.info(f"\nMoving to pose: {full_pose_action}")
        obs, reward, terminated = env.move_to_pose(full_pose_action)
        logger.info(
            f"Move result - EE Pose: {obs.get('ee_pos_w')}, Reward: {reward}, Terminated: {terminated}"
        )

        logger.info("\nClosing gripper...")
        obs, reward, terminated = env.close_gripper()
        logger.info(
            f"Close gripper result - Gripper Open: {obs.get('gripper_open')}, Reward: {reward}, Terminated: {terminated}"
        )

        if env.use_vlm and "Cube" in env.target_objects:
            logger.info("\nGetting 3D VLM observation for 'Cube'...")
            cube_3d_data = env.get_3d_obs_by_name_by_vlm("Cube")
            if cube_3d_data:
                for i, (points, normals) in enumerate(cube_3d_data):
                    logger.info(
                        f"  Cube Instance {i + 1}: {points.shape[0]} points, Normals shape: {normals.shape}"
                    )
            else:
                logger.info("  'Cube' not found by VLM or error occurred.")

        logger.info("\nResetting to default pose...")
        obs, reward, terminated = env.reset_to_default_pose()
        logger.info(f"Reset to default pose result - EE Pose: {obs.get('ee_pos_w')}")

    except Exception as e_main:
        logger.error(f"Main execution error: {e_main}", exc_info=True)
    # finally:
    # simulation_app.close() # Ensure simulation app is closed cleanly
    # logger.info("SimulationApp closed.")

    logger.info("Example script finished.")
