# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import inspect
from typing import Any, Dict, Optional
import gymnasium as gym
import os
import torch
import logging
import base64, requests
import numpy as np
import open3d as o3d
import logging
import time

import carb

from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport
from omni.kit.app import get_app

from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.data_collector import RobomimicDataCollector
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

from omni.isaac.lab_tasks.manager_based.manipulation.lift.config.franka.ik_rel_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
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
import omni.replicator.core as rep
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.xforms import reset_and_set_xform_ops
from pxr import Gf


from utils.decorator import async_to_sync, maybe_coroutine

logging.basicConfig(level=logging.INFO)


class Observation:
    """可自动处理协程的观察数据类"""

    def __init__(self):
        self._data: Dict[str, Any] = {}

    def update(self, other: Dict[str, Any]) -> None:
        """更新观察数据"""
        for k, v in other.items():
            if isinstance(v, dict):
                if k not in self._data:
                    self._data[k] = Observation()
                self._data[k].update(v)
            else:
                # 如果是协程对象，立即提交执行
                if inspect.iscoroutine(v):
                    from omni.kit.async_engine import run_coroutine

                    # try:
                    #     loop = asyncio.get_running_loop()
                    # except RuntimeError:
                    #     loop = asyncio.new_event_loop()
                    #     asyncio.set_event_loop(loop)
                    # task = loop.run_until_complete(v)
                    task = asyncio.ensure_future(v)
                    # task = run_coroutine(v)
                    self._data[k] = task  # 存储 Task 对象
                else:
                    self._data[k] = v

    def _getitem(self, key):
        """支持字典式访问并自动处理协程"""
        value = self._data[key]
        if isinstance(value, Observation):
            return value
        if isinstance(value, asyncio.Task):
            while not value.done():
                get_app().update()
                pass
            value = value.result()
        if isinstance(value, dict):
            obs = Observation()
            obs.update(value)
            value = obs
        return value

    def __getitem__(self, key):
        value = self._getitem(key)
        self._data[key] = value  # update coroutine obj to actual value
        return self._data[key]

    def __setitem__(self, key, value):
        """支持字典式赋值"""
        if isinstance(value, dict):
            obs = Observation()
            obs.update(value)
            self._data[key] = obs
        else:
            self._data[key] = value

    def get(self, key, default=None):
        """获取值，支持默认值"""
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key in self._data


class EnvIsaacLab:
    def __init__(self, task_name, cfg: dict, visualizer=None):
        env_cfg: FrankaCubeLiftEnvCfg = parse_env_cfg(task_name)  # type: ignore
        self.cfg = cfg

        env_cfg.terminations.time_out.time_out = False  # disable time out termination
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # set max_steps to large number to avoid termination
        env_cfg.episode_length_s = 1.0e9
        # we want to have the terms in the observations returned as a dictionary
        # rather than a concatenated tensor
        env_cfg.observations.policy.concatenate_terms = False
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.observations.rgb.concatenate_terms = False

        env_cfg.commands.object_pose.debug_vis = False  # disable debug visualization

        # create environment
        self.env = gym.make(task_name, cfg=env_cfg)

        # add people to env
        people_prim = add_reference_to_stage(
            usd_path=f"omniverse://localhost/Projects/lunarbase/chars/astro/astro.usd",
            prim_path="/World/House/People",
        )
        reset_and_set_xform_ops(
            prim=people_prim,
            translation=Gf.Vec3d(45.0, -70.0, 0.0),
            orientation=Gf.Quatd(0, 0, 0, 1.0),
            scale=Gf.Vec3d(100.0, 100.0, 100.0),
        )

        self._set_camera_view_to_target_object()
        self.env.unwrapped.sim.step()

        self.logger = logging.getLogger(__name__)

        self.target_objects = self.get_object_names()
        self.cameras = ["wrist", "front"]
        self.camera_info = {}
        self.category_multiplier = 100
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

        self.device = self.env.unwrapped.action_manager.device  # type: ignore
        self.latest_obs = Observation()
        self.init_obs = Observation()
        self.workspace_bounds_min = np.array([-0.1, -0.5, 0])
        self.workspace_bounds_max = np.array([1.0, 0.5, 1.0])
        self.visualizer = visualizer
        if self.visualizer is not None:
            self.visualizer.update_bounds(
                self.workspace_bounds_min, self.workspace_bounds_max
            )
        self.use_vlm = True

    def turn_off_vlm(self):
        self.use_vlm = False

    def _set_camera_view_to_target_object(self):
        viewport = get_active_viewport()
        # viewport.set_active_camera(path)
        set_camera_view(
            eye=np.array([1.6, 0.0, 0.6]),
            target=np.array([0, 0, 0]),
            viewport_api=viewport,
        )

    async def _request_process_sole_frame(self, labels, data_array):
        url = "http://127.0.0.1:5000/process_sole_frame"
        data_bytes = data_array.tobytes()
        data_base64 = base64.b64encode(data_bytes).decode("utf-8")

        data = {"label": labels, "data": data_base64, "shape": data_array.shape}

        resp = requests.post(url, json=data)

        if resp.status_code == 200:
            result = resp.json()
            return np.array(result).astype(np.uint32)
        else:
            print(
                f"Request process_sole_frame failed with status code {resp.status_code}"
            )
            return None

    async def _request_process_first_frame(self, labels, data_array):
        url = "http://127.0.0.1:5000/process_first_frame"
        data_bytes = data_array.tobytes()
        data_base64 = base64.b64encode(data_bytes).decode("utf-8")

        data = {"label": labels, "data": data_base64, "shape": data_array.shape}

        resp = requests.post(url, json=data)

        if resp.status_code == 200:
            result = resp.json()
            return np.array(result).astype(np.uint32)
        else:
            print(
                f"Request process_first_frame failed with status code {resp.status_code}"
            )
            return None

    async def _request_process_frame(self, cam_obs_task: asyncio.Task):
        url = "http://127.0.0.1:5000/process_frame"

        cam_obs = await cam_obs_task  # await

        rgbs = [cam_obs[f"{cam}_rgb"] for cam in self.cameras]
        rgbs = np.stack(list(rgbs), axis=0)

        data_bytes = rgbs.tobytes()
        data_base64 = base64.b64encode(data_bytes).decode("utf-8")

        data = {"data": data_base64, "shape": rgbs.shape}

        resp = requests.post(url, json=data)

        if resp.status_code == 200:
            result = resp.json()
            cam_obs.update({"masks": np.array(result).astype(np.uint32)})
        else:
            print(f"Request process_frame failed with status code {resp.status_code}")
        return cam_obs

    def reset(self):
        obs_dict = {}

        obs, _ = self.env.reset()

        ee_pos_w = obs["policy"]["ee_pos"][0][0]
        ee_quat_w = obs["policy"]["ee_quat"][0][0]

        obs_dict.update({"ee_pos_w": ee_pos_w.cpu().numpy()})
        obs_dict.update({"ee_quat_w": ee_quat_w.cpu().numpy()})

        joint_pos_norm = obs["policy"]["joint_pos_norm"][0]
        ee_joint_indices = (
            self.env.unwrapped.scene["robot"].actuators["panda_hand"].joint_indices  # type: ignore
        )
        ee_joint_pos = joint_pos_norm[ee_joint_indices]
        if all(ee_joint_pos > 0.9):
            obs_dict.update({"gripper_open": 1.0})
        else:
            obs_dict.update({"gripper_open": 0.0})

        cam_obs = Observation()
        cam_obs.update({"cam_obs": self.get_camera_observation(self.env.unwrapped)})  # type: ignore
        # print(f"cam_obs async Task name: {cam_obs._data['cam_obs'].get_name()}")
        cam_obs = cam_obs["cam_obs"]  # await
        if self.use_vlm:
            rgbs = [cam_obs[f"{cam}_rgb"] for cam in self.cameras]
            rgbs = np.stack(list(rgbs), axis=0)
            masks = self._request_process_first_frame(
                labels=self.target_objects, data_array=rgbs
            )
            cam_obs.update({"masks": masks})
        obs_dict.update({"camera": cam_obs})
        self.latest_obs.update(obs_dict)
        self.init_obs.update(obs_dict)
        if self.use_vlm and self.latest_obs["camera"]["masks"] is None:
            carb.log_error("Failed to process first frame.")
            raise ValueError("Failed to process first frame.")
        self.latest_action = None
        return obs_dict

    def get_object_names(self):
        return self.cfg["scene_target_objects"]

    def get_frame_transforms(
        self,
        frame_name,
        robot,
        rotation,
        translation,
        relative_translation: bool = True,
    ):
        """
        rotation: torch.Tensor, shape=(4,), dtype=torch.float, w.r.t. the relative frame (i.e. last frame)
        translation: torch.Tensor, shape=(3,), dtype=torch.float, w.r.t. the relative frame (i.e. last frame)
        """
        frame_idx = [
            idx for idx, name in enumerate(robot.body_names) if name == frame_name
        ][0]
        frame_pos = robot.data.body_pos_w[:, frame_idx, :].cpu()
        frame_quat = robot.data.body_quat_w[:, frame_idx, :].cpu()
        offset = torch.tensor(
            translation, dtype=frame_pos.dtype, device=frame_pos.device
        ).expand_as(frame_pos)
        trans_quat = quat_mul(
            frame_quat, rotation
        )  # 右乘为 inner rotation (i.e. body-fixed rotation), 左乘为 outer rotation (i.e. space-fixed rotation)
        if relative_translation:
            trans_pos = frame_pos + quat_rotate(
                trans_quat, offset
            )  # 先将 offset 旋转到 frame 的坐标系下，再加上 frame 的位置
        else:
            trans_pos = frame_pos + offset
        return trans_pos, trans_quat

    def _build_extrinsic_matrix(
        self, position: torch.Tensor, orientation: torch.Tensor
    ):
        assert position.shape == (3,)
        assert orientation.shape == (4,)

        R = matrix_from_quat(orientation)
        extrinsic_matrix = torch.eye(4, dtype=torch.float)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = position

        return extrinsic_matrix

    async def get_camera_observation(self, env: ManagerBasedEnv):
        """F
        Due to current limitations in the renderer, we can have only one TiledCamera instance in the scene. For use cases that require a setup with more than one camera, we can imitate the multi-camera behavior by moving the location of the camera in between render calls in a step.
        For example, in a stereo vision setup, the below snippet can be implemented:

        # render image from "first" camera
        camera_data_1 = self._tiled_camera.data.output["rgb"].clone() / 255.0
        # update camera transform to the "second" camera location
        self._tiled_camera.set_world_poses(
            positions=pos,
            orientations=rot,
            convention="world"
        )
        # step the renderer
        self.sim.render()
        self._tiled_camera.update(0, force_recompute=True)
        # render image from "second" camera
        camera_data_2 = self._tiled_camera.data.output["rgb"].clone() / 255.0

        As for robot env, we should trans front tiled camera (default position) to the wrist of the robot and render the image.

        - :obj:`"opengl"` - forward axis: -Z - up axis +Y - Offset is applied in the OpenGL (Usd.Camera) convention
        - :obj:`"ros"`    - forward axis: +Z - up axis -Y - Offset is applied in the ROS convention
        - :obj:`"world"`  - forward axis: +X - up axis +Z - Offset is applied in the World Frame convention

        default camera axis: -Z (forward), +Y (up), +X (right)
        """

        carb.log_info("Start async get_camera_observation")

        matrix_from_ros_to_world = torch.Tensor([[[0, 0, 1], [-1, 0, 0], [0, -1, 0]]])
        q_z_n90 = quat_from_angle_axis(
            torch.tensor([-0.5 * torch.pi]), torch.tensor([0, 0, 1], dtype=torch.float)
        )
        q_z_90 = quat_from_angle_axis(
            torch.tensor([0.5 * torch.pi]), torch.tensor([0, 0, 1], dtype=torch.float)
        )
        q_x_180 = quat_from_angle_axis(
            torch.tensor([torch.pi]), torch.tensor([1, 0, 0], dtype=torch.float)
        )
        q_y_60 = quat_from_angle_axis(
            torch.tensor([1 / 3 * torch.pi]), torch.tensor([0, 1, 0], dtype=torch.float)
        )
        q_y_30 = quat_from_angle_axis(
            torch.tensor([1 / 6 * torch.pi]), torch.tensor([0, 1, 0], dtype=torch.float)
        )

        camera_trans_dict = {
            "wrist": self.get_frame_transforms(
                "panda_hand",
                robot := env.scene.articulations.get("robot"),  # type: ignore
                rotation=quat_mul(q_x_180, q_z_n90),
                translation=torch.tensor([0.0, 0.05, -0.06], dtype=torch.float),
            ),
            "front": self.get_frame_transforms(
                "panda_link0",
                robot := env.scene.articulations.get("robot"),  # type: ignore
                rotation=quat_mul(q_y_60, q_z_90),  # inner rotation: y_60 -> z_90
                translation=torch.tensor([1.6, 0.0, 0.6], dtype=torch.float),
                relative_translation=False,
            ),
        }

        world_lcoation = torch.tensor([0, 0, 0])
        forward_vector = torch.tensor([0, 0, -1.0])
        wrist_cam_location = camera_trans_dict["wrist"][0].squeeze()
        front_cam_location = camera_trans_dict["front"][0].squeeze()

        world_rotation = torch.tensor([1, 0, 0, 0], dtype=torch.float)
        wrist_cam_rotation = camera_trans_dict["wrist"][1].squeeze()
        front_cam_rotation = camera_trans_dict["front"][1].squeeze()

        camera: TiledCamera = env.scene.sensors["tiled_camera"]  # type: ignore
        camera_data_dict = {}
        for key, value in camera_trans_dict.items():
            # 设置相机位置和方向
            camera.set_world_poses(
                positions=value[0], orientations=value[1], convention="opengl"
            )
            # 渲染新视角
            await env.sim.render_async()
            await get_app().next_update_async()
            camera.update(0, force_recompute=True)

            intrinsic_matrices = camera.data.intrinsic_matrices.clone()
            extrinsic_matrice = self._build_extrinsic_matrix(
                position=value[0].squeeze(), orientation=value[1].squeeze()
            )
            lookat = extrinsic_matrice[:3, :3] @ forward_vector
            self.camera_info.update({f"{key}_extrinsic_matrice": extrinsic_matrice})
            self.camera_info.update({f"{key}_intrinsic_matrices": intrinsic_matrices})
            self.camera_info.update({f"{key}_lookat": lookat})
            # 获取并处理图像数据
            rgb_data = camera.data.output["rgb"].clone()  # torch.Size([1, 256, 256, 3])
            depth_data = camera.data.output[
                "distance_to_image_plane"
            ].clone()  # torch.Size([1, 256, 256, 1])
            pointcloud_data = create_pointcloud_from_depth(
                intrinsic_matrix=intrinsic_matrices.squeeze(0),  # torch.Size([3, 3])
                depth=depth_data.squeeze(0).squeeze(-1),  # torch.Size([256, 256])
                orientation=q_x_180,  # transform to camera coordinate (not camera optical coordinate)
                keep_invalid=True,
            )
            pointcloud_data = transform_points(
                pointcloud_data,
                orientation=value[1].squeeze().tolist(),
                position=value[0].squeeze().tolist(),
            )  # transform to world coordinate
            # save_images_to_file(rgb_data / 255, f"camera_rgb_{key}.png")
            # save_images_to_file(
            #     (depth_data.clone() - depth_data.min())
            #     / (depth_data.max() - depth_data.min()),
            #     f"camera_depth_{key}.png",
            # )
            # np.savetxt(
            #     f"camera_pointcloud_{key}.txt",
            #     pointcloud_data.numpy(force=True),  # type: ignore
            #     delimiter=",",
            # )
            camera_data_dict[f"{key}_rgb"] = (
                rgb_data[0].cpu().permute(2, 0, 1).numpy()
            )  # H x W x C --> C x H x W
            camera_data_dict[f"{key}_depth"] = depth_data[0].cpu().numpy()
            camera_data_dict[f"{key}_pointcloud"] = (
                pointcloud_data.cpu().numpy()
                if type(pointcloud_data) is torch.Tensor
                else pointcloud_data
            )
        carb.log_info("End async get_camera_observation")
        return camera_data_dict

    def get_3d_obs_by_name_by_vlm(self, query_name):
        assert self.use_vlm, "VLM is not enabled."

        if not self.latest_obs:
            carb.log_error("No observation available.")
            raise ValueError("No observation available.")

        assert query_name in self.target_objects, f"Unknown object name: {query_name}"

        latest_masks: Optional[np.ndarray] = self.latest_obs["camera"]["masks"]
        if latest_masks is None:
            carb.log_error("No masks available.")
            raise ValueError("No masks available.")  # TODO need to fallback
        latest_masks = latest_masks.transpose(0, 2, 1)  # keep first dimension
        points, masks, normals = [], [], []
        for idx, cam in enumerate(self.cameras):
            point = self.latest_obs["camera"][f"{cam}_pointcloud"]
            mask_frame = latest_masks[idx]
            lookat_vector = self.camera_info[f"{cam}_lookat"]
            # save mask as image
            # import cv2

            # cv2.imwrite(f"{cam}_mask.png", mask_frame.astype(np.uint8))
            # np.savetxt(f"{cam}_pointcloud.txt", point, delimiter=",")
            # np.savetxt(f"{cam}_pointcloud_part.txt", point[0:1000, :], delimiter=",")
            # np.savetxt(
            #     f"{cam}_masked_pointcloud.txt",
            #     point[mask_frame.reshape(-1) > 0, :],
            #     delimiter=",",
            # )
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
            flip_indices = np.dot(cam_normals, lookat_vector) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)
            # break  # for test
        points = np.concatenate(points, axis=0)
        masks = np.concatenate(masks, axis=0)  # [0,101,102,201,202,0,0,0,301]
        categery_masks = (
            masks.astype(np.int32) // self.category_multiplier
        )  # [0,1,1,2,2,0,0,0,3]
        normals = np.concatenate(normals, axis=0)
        # get object points
        category_label = self.name2categerylabel[query_name]  # 1
        # objs_mask: [0,101,102,0,0,0,0,0,0]
        masks[~np.isin(categery_masks, category_label)] = 0  # [0,101,102,0,0,0,0,0,0]
        if not np.any(masks):
            # which masks == [0,0,0,0,0,0,0,0,0] if category_label == 4
            self.logger.warning(f"Object {query_name} not found in the scene")
            return None
        # remove the background # [1,2] which measn there are two instances of this object
        object_instance_label = np.unique(np.mod(masks, self.category_multiplier))[1:]
        assert len(object_instance_label) > 0, (
            f"Object {query_name} not found in the scene"
        )
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
            # np.savetxt(f"{query_name}_{obj_ins_id}_points.txt", obj_points, delimiter=",")
            obj_normals = np.asarray(pcd_downsampled_filted.normals)
            objs_points.append(obj_points)
            objs_normals.append(obj_normals)

        print(f"we find {len(objs_points)} instances of {query_name}")
        return zip(objs_points, objs_normals)

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        points = []
        for cam in self.cameras:
            point = self.latest_obs["camera"][f"{cam}_pointcloud"]
            points.append(point.reshape(-1, 3))
        points = np.concatenate(points, axis=0)
        return points, None

    def apply_action(self, action, relative_mode=False):
        """
        Applies an action in the environment and updates the state.

        Args:
            action: The action to apply.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        import time

        t0 = time.time()
        action = self._process_action(action)
        delta = 0.1
        close = False
        # if the gripper is open and the action contains a close gripper command,
        # temporarily disable the gripper, close it after moving
        if (
            relative_mode is False
            and torch.any(action[..., -1] < 0.0)
            and self.latest_obs["gripper_open"] == 1.0
        ):
            close = True
            action[..., -1] = (
                0.0  # temporarily disable the gripper, close it after moving
            )
            delta = 0.02  # improve the accuracy of the gripper as we need to close the gripper
            logging.info("Gripper will be closed after moving.")
        print(f"time (_process_action): {time.time() - t0}")
        t0 = time.time()
        while True:  # step until the moving action is done
            obs, reward, terminate, truncated, info = self.env.step(
                action
            )  # action input quat shape is wxyz
            self.latest_obs.update(self._process_obs(obs, do_vlm=False))
            self.latest_reward = reward
            self.latest_terminate = terminate
            self.latest_action = action.squeeze(0).cpu().numpy()
            if (
                relative_mode is False
                and np.sum(np.abs(self.get_ee_pose()[:3] - self.latest_action[:3]))
                < delta
            ) or (relative_mode is True and time.time() - t0 > 1 / 3):
                # absolute mode: check the position of the end effector
                # relative mode: check the time: 3Hz
                break
            self.logger.info(
                f"delta: {np.sum(np.abs(self.get_ee_pose() - self.latest_action[:7]))}"
            )
        print(f"time (stepping): {time.time() - t0}")
        t0 = time.time()
        if relative_mode is False and close:
            # now we can close the gripper
            action[..., -1] = -1.0  # recover the close gripper command
            self.logger.info("Closing gripper.")
            # loop until the gripper is closed
            while self.latest_obs["gripper_open"] == 1.0:
                obs, reward, terminate, truncated, info = self.env.step(
                    action
                )  # action input quat shape is wxyz
                self.latest_obs.update(self._process_obs(obs, do_vlm=False))
                self.latest_reward = reward
                self.latest_terminate = terminate
                self.latest_action = action.squeeze(0).cpu().numpy()

            # more steps to ensure the gripper is closed
            for i in range(10):
                obs, reward, terminate, truncated, info = self.env.step(
                    action
                )  # action input quat shape is wxyz
                self.latest_obs.update(self._process_obs(obs, do_vlm=False))
                self.latest_reward = reward
                self.latest_terminate = terminate
        print(f"time (close): {time.time() - t0}")
        t0 = time.time()
        self.latest_obs.update(self._process_obs(obs, do_vlm=True))
        print(f"time (_process_obs): {time.time() - t0}")
        t0 = time.time()
        return self.latest_obs, reward, terminate

    def move_to_pose(self, pose, speed=None):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            speed: The speed at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([pose, [self.init_obs["gripper_open"]]])
        else:
            action = np.concatenate([pose, [self.latest_action[-1]]])
        return self.apply_action(action)

    def open_gripper(self):
        """
        Opens the gripper of the robot.
        """
        action = np.concatenate(
            [self.latest_obs["ee_pos_w"], self.latest_obs["ee_quat_w"], [1.0]]
        )
        return self.apply_action(action)

    def close_gripper(self):
        """
        Closes the gripper of the robot.
        """
        action = np.concatenate(
            [self.latest_obs["ee_pos_w"], self.latest_obs["ee_quat_w"], [0.0]]
        )
        return self.apply_action(action)

    def set_gripper_state(self, gripper_state):
        """
        Sets the state of the gripper.

        Args:
            gripper_state: The target state for the gripper.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        action = np.concatenate(
            [self.latest_obs["ee_pos_w"], self.latest_obs["ee_quat_w"], [gripper_state]]
        )
        return self.apply_action(action)

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate(
                [
                    self.init_obs["ee_pos_w"],
                    self.init_obs["ee_quat_w"],
                    [self.init_obs["gripper_open"]],
                ]
            )
        else:
            action = np.concatenate(
                [
                    self.init_obs["ee_pos_w"],
                    self.init_obs["ee_quat_w"],
                    [self.latest_action[-1]],
                ]
            )
        return self.apply_action(action)

    def get_ee_pose(self):
        assert self.latest_obs is not None, "Please reset the environment first"
        return np.concatenate(
            [self.latest_obs["ee_pos_w"], self.latest_obs["ee_quat_w"]]
        )

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[3:]

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

    def _process_obs(self, obs, do_vlm=True):
        obs_dict = {}

        ee_pos_w = obs["policy"]["ee_pos"][0][0]
        ee_quat_w = obs["policy"]["ee_quat"][0][0]

        obs_dict.update({"ee_pos_w": ee_pos_w.cpu().numpy()})
        obs_dict.update({"ee_quat_w": ee_quat_w.cpu().numpy()})

        joint_pos_norm = obs["policy"]["joint_pos_norm"][0]
        ee_joint_indices = (
            self.env.unwrapped.scene["robot"].actuators["panda_hand"].joint_indices  # type: ignore
        )
        ee_joint_pos = joint_pos_norm[ee_joint_indices]
        if all(ee_joint_pos > 0.9):
            obs_dict.update({"gripper_open": 1.0})
        else:
            obs_dict.update({"gripper_open": 0.0})
        if self.use_vlm and do_vlm:
            cam_obs_task = self.get_camera_observation(self.env.unwrapped)
            t0 = time.time()
            cam_obs_async = self._request_process_frame(cam_obs_task)
            print(f"time (_request_process_frame): {time.time() - t0}")
            t0 = time.time()
            obs_dict.update({"camera": cam_obs_async})
            print(f"time (update): {time.time() - t0}")
            t0 = time.time()
        else:
            cam_obs = Observation()
            cam_obs.update({"cam_obs": self.get_camera_observation(self.env.unwrapped)})  # type: ignore
            # print(f"cam_obs async Task name: {cam_obs._data['cam_obs'].get_name()}")
            cam_obs = cam_obs["cam_obs"]  # await
            obs_dict.update({"camera": cam_obs})
        return obs_dict

    def _process_action(self, action):
        if action[-1] < 1.0:
            action[-1] = -1.0
        action = torch.tensor(action, dtype=torch.float, device=self.device).repeat(
            1, 1
        )
        return action


if __name__ == "__main__":
    env = EnvIsaacLab(task_name=args_cli.task)
    obs = env.reset()
