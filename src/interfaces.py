from typing import Union, Callable
from LMP import LMP
from utils import (
    get_clock_time,
    normalize_vector,
    pointat2quat,
    bcolors,
    Observation,
    VoxelIndexingWrapper,
)
import numpy as np
from planners import PathPlanner
import time
from scipy.ndimage import distance_transform_edt
import transforms3d
from controllers import Controller, SimpleQuadcopterController, SimpleROSController
import controllers

from envs.pyrep_env.pyrep_quad_env import VoxPoserPyRepQuadcopterEnv
from envs.ros_env.ros_env import VoxPoserROSDroneEnv
from envs.dummy_env import DummyEnv

# creating some aliases for end effector and table in case LLMs refer to them differently (but rarely this happens)
EE_ALIAS = [
    "ee",
    "endeffector",
    "end_effector",
    "end effector",
    "gripper",
    "hand",
    "copter",
    "quadricopter",
    "UAV",
    "quadcopter",
    "controlled object",
]
TABLE_ALIAS = [
    "table",
    "desk",
    "workstation",
    "work_station",
    "work station",
    "workspace",
    "work_space",
    "work space",
]


class LMP_interface:
    def __init__(
        self,
        env: Union[VoxPoserPyRepQuadcopterEnv, VoxPoserROSDroneEnv, DummyEnv],
        lmp_config,
        controller_config,
        planner_config,
        env_name="pyrep_quadcopter",
    ):
        self._env = env
        self._env_name = env_name
        self._cfg = lmp_config
        self._map_size = self._cfg["map_size"]
        self._planner = PathPlanner(planner_config, map_size=self._map_size)
        _controller_type = controller_config["type"]
        if _controller_type == "SimpleQuadcopterController":
            self._controller = SimpleQuadcopterController(self._env, controller_config)
        elif _controller_type == "SimpleROSController":
            self._controller = SimpleROSController(self._env, controller_config)
        else:
            self._controller = Controller(self._env, controller_config)
        self.is_quad_env = True
        # calculate size of each voxel (resolution)
        self._resolution = (
            self._env.workspace_bounds_max - self._env.workspace_bounds_min
        ) / self._map_size
        print("#" * 50)
        print(f"## voxel resolution: {self._resolution}")
        print("#" * 50)
        print()
        print()

    # ======================================================
    # == functions exposed to LLM
    # ======================================================
    def get_ee_pos(self):
        return self._world_to_voxel(self._env.get_ee_pos())

    def detect(self, obj_name):
        return self.detect_quad(obj_name)

    def execute(
        self,
        movable_obs_func,
        affordance_map=None,
        avoidance_map=None,
        velocity_map=None,
        rotation_map=None,
        gripper_map=None,
    ):
        return self.execute_quad(
            movable_obs_func,
            affordance_map,
            avoidance_map,
            velocity_map,
            rotation_map,
            gripper_map,
        )

    def detect_quad(self, obj_name):
        """return an observation dict containing useful information about the object"""
        assert self._env.vlm is not None, "please enable VLM before calling detect_quad"
        object_obs_list = []
        # enable_vlm = False
        print(f"[interface.py] calling VLM to detect {obj_name}")
        if obj_name.lower() in EE_ALIAS:
            # 如果是执行器则不需要调用模型进行检测
            print(
                f"[interface.py:detect_quad] as {obj_name} is the end effector, returning it directly"
            )
            obs_dict = dict()
            obs_dict["name"] = obj_name
            obs_dict["position"] = self.get_ee_pos()
            obs_dict["aabb"] = np.array([self.get_ee_pos(), self.get_ee_pos()])
            obs_dict["_position_world"] = self._env.get_ee_pos()

            return [Observation(obs_dict)]
        else:
            if type(self._env) == DummyEnv:
                return {
                    "name": "dummy_object",
                    "position": np.array([0, 0, 0]),
                    "aabb": np.array([[0, 0, 0], [0, 0, 0]]),
                    "_position_world": np.array([0, 0, 0]),
                }
            obs_results = self._env.get_3d_obs_by_name_by_vlm(obj_name)
            if (
                obs_results is None
            ):  # actually if obs_results is None, it would raise an error in the get_3d_obs_by_name_by_vlm function or make quadcopter to fly higher?
                return object_obs_list
            for id, obs_result in enumerate(obs_results):
                obs_dict = dict()
                obj_pc, obj_normal = obs_result
                if obj_pc.size == 0:
                    print(
                        f"[interface.py:detect_quad] detected empty point cloud for {obj_name} (id: {id})"
                    )
                    continue
                voxel_map = self._points_to_voxel_map(obj_pc)  # 体素图
                aabb_min = self._world_to_voxel(
                    np.min(obj_pc, axis=0)
                )  # 体素图中每个坐标轴的最小坐标 (x_min, y_min, z_min)
                aabb_max = self._world_to_voxel(
                    np.max(obj_pc, axis=0)
                )  # 体素图中每个坐标轴的最大坐标 (x_max, y_max, z_max)
                obs_dict["occupancy_map"] = voxel_map  # in voxel frame
                obs_dict["name"] = obj_name
                obs_dict["position"] = self._world_to_voxel(
                    np.mean(obj_pc, axis=0)
                )  # in voxel frame 用点云的均值代表物体的位置
                obs_dict["aabb"] = np.array([aabb_min, aabb_max])  # in voxel frame
                obs_dict["_position_world"] = np.mean(obj_pc, axis=0)  # in world frame
                obs_dict["_point_cloud_world"] = obj_pc  # in world frame
                obs_dict["normal"] = normalize_vector(obj_normal.mean(axis=0))
                object_obs_list.append(Observation(obs_dict))
        return object_obs_list

    def execute_quad(
        self,
        movable_obs_func: Callable,
        affordance_map: Callable = None,
        avoidance_map: Callable = None,
        velocity_map: Callable = None,
        rotation_map: Callable = None,
        gripper_map: Callable = None,
    ):
        """
        First use planner to generate waypoint path, then use controller to follow the waypoints.

        Args:
          movable_obs_func: callable function to get observation of the body to be moved
          affordance_map: callable function that generates a 3D numpy array, the target voxel map
          avoidance_map: callable function that generates a 3D numpy array, the obstacle voxel map
          rotation_map: callable function that generates a 4D numpy array, the rotation voxel map (rotation is represented by a quaternion *in world frame*), not used for quadricopter
          velocity_map: callable function that generates a 3D numpy array, the velocity voxel map
          gripper_map: callable function that generates a 3D numpy array, the gripper voxel map, not used for quadricopter
        """
        print("[interface.py] calling execute")
        assert gripper_map is None, "gripper_map is not used for quadricopter"
        assert (
            rotation_map is None
        ), "rotation_map is not used for quadricopter as we havent implemented it"
        # initialize default voxel maps if not specified
        if velocity_map is None:
            velocity_map = self._get_default_voxel_map("velocity")
        if avoidance_map is None:
            avoidance_map = self._get_default_voxel_map("obstacle")

        object_centric = (
            not movable_obs_func()["name"] in EE_ALIAS
        )  # false as the object is the quadricopter
        assert not object_centric, "the movable object must be the quadricopter"
        execute_info = []
        if affordance_map is not None:
            # execute path in closed-loop
            print(f"[interface.py] max plan iter: {self._cfg['max_plan_iter']}")
            for plan_iter in range(self._cfg["max_plan_iter"]):
                step_info = dict()
                # evaluate voxel maps such that we use latest information
                # 即：假设在execute的时候环境已经发生了变化，那么这里就会重新计算一次voxel map
                # 在执行路径的时候，不会对voxel map进行修改，也就是说不会调用detect函数来检测物体
                movable_obs = movable_obs_func()
                _affordance_map = affordance_map()
                _pre_avoidance_map = avoidance_map()
                _velocity_map = velocity_map()
                # TODO preprocess avoidance map, havent implemented yet
                _avoidance_map = self._preprocess_avoidance_map(
                    _pre_avoidance_map, _affordance_map, movable_obs
                )
                # start planning
                start_pos = movable_obs["position"]
                start_time = time.time()
                # optimize path and log
                path_voxel, planner_info = self._planner.optimize(
                    start_pos,
                    _affordance_map,
                    _avoidance_map,
                    object_centric=object_centric,
                )
                print(
                    f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] planner time: {time.time() - start_time:.3f}s{bcolors.ENDC}"
                )
                assert len(path_voxel) > 0, "path_voxel is empty"
                step_info["path_voxel"] = path_voxel
                step_info["planner_info"] = planner_info
                # convert voxel path to world trajectory, and include rotation, velocity, and gripper information
                traj_world = self._path2traj(
                    path_voxel,
                    rotation_map=None,
                    velocity_map=_velocity_map,
                    gripper_map=None,
                )  # we will get a list of tuple: (world_xyz, rotation, velocity, gripper)
                traj_world = traj_world[: self._cfg["num_waypoints_per_plan"]]
                step_info["start_pos"] = start_pos
                step_info["plan_iter"] = plan_iter
                step_info["movable_obs"] = movable_obs
                step_info["traj_world"] = traj_world
                step_info["affordance_map"] = _affordance_map
                step_info["rotation_map"] = (
                    rotation_map() if rotation_map is not None else None
                )
                step_info["velocity_map"] = _velocity_map
                step_info["gripper_map"] = (
                    gripper_map() if gripper_map is not None else None
                )
                step_info["avoidance_map"] = _avoidance_map
                step_info["pre_avoidance_map"] = _pre_avoidance_map

                # visualize
                if self._cfg["visualize"]:
                    assert self._env.visualizer is not None
                    step_info["start_pos_world"] = self._voxel_to_world(start_pos)
                    step_info["targets_world"] = self._voxel_to_world(
                        planner_info["targets_voxel"]
                    )
                    self._env.visualizer.visualize(step_info)

                # execute path
                print(
                    f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] start executing path via controller ({len(traj_world)} waypoints){bcolors.ENDC}"
                )
                controller_infos = dict()
                for i, waypoint in enumerate(traj_world):
                    # check if the movement is finished
                    if (
                        np.linalg.norm(
                            movable_obs["_position_world"] - traj_world[-1][0]
                        )
                        <= 0.01
                    ):
                        print(
                            f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached last waypoint; curr_xyz={movable_obs['_position_world']}, target={traj_world[-1][0]} (distance: {np.linalg.norm(movable_obs['_position_world'] - traj_world[-1][0]):.3f})){bcolors.ENDC}"
                        )
                        break
                    # skip waypoint if moving to this point is going in opposite direction of the final target point
                    # (for example, if you have over-pushed an object, no need to move back)
                    if i != 0 and i != len(traj_world) - 1:
                        movable2target = (
                            traj_world[-1][0] - movable_obs["_position_world"]
                        )
                        movable2waypoint = waypoint[0] - movable_obs["_position_world"]
                        if np.dot(movable2target, movable2waypoint).round(3) <= 0:
                            print(
                                f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] skip waypoint {i+1} because it is moving in opposite direction of the final target{bcolors.ENDC}"
                            )
                            continue
                    # execute waypoint
                    controller_info = self._controller.execute(movable_obs, waypoint)
                    # loggging
                    movable_obs = movable_obs_func()
                    dist2target = np.linalg.norm(
                        movable_obs["_position_world"] - traj_world[-1][0]
                    )
                    if not object_centric and controller_info["mp_info"] == -1:
                        print(
                            f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] failed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_obs["_position_world"].round(3)}, target: {traj_world[-1][0].round(3)}, start: {traj_world[0][0].round(3)}, dist2target: {dist2target.round(3)}); mp info: {controller_info["mp_info"]}{bcolors.ENDC}'
                        )
                    else:
                        print(
                            f'{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] completed waypoint {i+1} (wp: {waypoint[0].round(3)}, actual: {movable_obs["_position_world"].round(3)}, target: {traj_world[-1][0].round(3)}, start: {traj_world[0][0].round(3)}, dist2target: {dist2target.round(3)}){bcolors.ENDC}'
                        )
                    controller_info["controller_step"] = i
                    controller_info["target_waypoint"] = waypoint
                    controller_infos[i] = controller_info
                step_info["controller_infos"] = controller_infos
                execute_info.append(step_info)
                # check whether we need to replan
                curr_pos = movable_obs["position"]
                if distance_transform_edt(1 - _affordance_map)[tuple(curr_pos)] <= 2:
                    print(
                        f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] reached target; terminating {bcolors.ENDC}"
                    )
                    break
        print(
            f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] finished executing path via controller{bcolors.ENDC}"
        )
        self._env.visualizer.save_gifs()
        return execute_info

    def cm2index(self, cm, direction):
        if isinstance(direction, str) and direction == "x":
            x_resolution = self._resolution[0] * 100  # resolution is in m, we need cm
            return int(cm / x_resolution)
        elif isinstance(direction, str) and direction == "y":
            y_resolution = self._resolution[1] * 100
            return int(cm / y_resolution)
        elif isinstance(direction, str) and direction == "z":
            z_resolution = self._resolution[2] * 100
            return int(cm / z_resolution)
        else:
            # calculate index along the direction
            assert isinstance(direction, np.ndarray) and direction.shape == (3,)
            direction = normalize_vector(direction)
            x_cm = cm * direction[0]
            y_cm = cm * direction[1]
            z_cm = cm * direction[2]
            x_index = self.cm2index(x_cm, "x")
            y_index = self.cm2index(y_cm, "y")
            z_index = self.cm2index(z_cm, "z")
            return np.array([x_index, y_index, z_index])

    def index2cm(self, index, direction=None):
        if direction is None:
            average_resolution = np.mean(self._resolution)
            return index * average_resolution * 100  # resolution is in m, we need cm
        elif direction == "x":
            x_resolution = self._resolution[0] * 100
            return index * x_resolution
        elif direction == "y":
            y_resolution = self._resolution[1] * 100
            return index * y_resolution
        elif direction == "z":
            z_resolution = self._resolution[2] * 100
            return index * z_resolution
        else:
            raise NotImplementedError

    def pointat2quat(self, vector):
        assert isinstance(vector, np.ndarray) and vector.shape == (
            3,
        ), f"vector: {vector}"
        return pointat2quat(vector)

    def set_voxel_by_radius(self, voxel_map, voxel_xyz, radius_cm=0, value=1):
        """given a 3D np array, set the value of the voxel at voxel_xyz to value. If radius is specified, set the value of all voxels within the radius to value."""
        voxel_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]] = value
        if type(radius_cm) != list and radius_cm > 0:
            radius_x = self.cm2index(radius_cm, "x")
            radius_y = self.cm2index(radius_cm, "y")
            radius_z = self.cm2index(radius_cm, "z")
            # simplified version - use rectangle instead of circle (because it is faster)
            min_x = max(0, voxel_xyz[0] - radius_x)
            max_x = min(self._map_size, voxel_xyz[0] + radius_x + 1)
            min_y = max(0, voxel_xyz[1] - radius_y)
            max_y = min(self._map_size, voxel_xyz[1] + radius_y + 1)
            min_z = max(0, voxel_xyz[2] - radius_z)
            max_z = min(self._map_size, voxel_xyz[2] + radius_z + 1)
            voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
        elif type(radius_cm) == list:
            assert len(radius_cm) == 3, "radius_cm must be a list of length 3"
            radius_x = self.cm2index(radius_cm[0], "x")
            radius_y = self.cm2index(radius_cm[1], "y")
            radius_z = self.cm2index(radius_cm[2], "z")
            # simplified version - use rectangle instead of circle (because it is faster)
            min_x = max(0, voxel_xyz[0] - radius_x)
            max_x = min(self._map_size, voxel_xyz[0] + radius_x + 1)
            min_y = max(0, voxel_xyz[1] - radius_y)
            max_y = min(self._map_size, voxel_xyz[1] + radius_y + 1)
            min_z = max(0, voxel_xyz[2] - radius_z)
            max_z = min(self._map_size, voxel_xyz[2] + radius_z + 1)
            voxel_map[min_x:max_x, min_y:max_y, min_z:max_z] = value
        return voxel_map

    def get_empty_affordance_map(self):
        return self._get_default_voxel_map(
            "target"
        )()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)

    def get_empty_avoidance_map(self):
        return self._get_default_voxel_map(
            "obstacle"
        )()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)

    def get_empty_rotation_map(self):
        return self._get_default_voxel_map(
            "rotation"
        )()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)

    def get_empty_velocity_map(self):
        return self._get_default_voxel_map(
            "velocity"
        )()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)

    def get_empty_gripper_map(self):
        return self._get_default_voxel_map(
            "gripper"
        )()  # return evaluated voxel map instead of functions (such that LLM can manipulate it)

    def reset_to_default_pose(self):
        self._env.reset_to_default_pose()

    # ======================================================
    # == helper functions
    # ======================================================
    def _world_to_voxel(self, world_xyz):
        _world_xyz = world_xyz.astype(np.float32)
        _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
        _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
        _map_size = self._map_size
        voxel_xyz = pc2voxel(
            _world_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size
        )
        return voxel_xyz

    def _voxel_to_world(self, voxel_xyz):
        _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
        _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
        _map_size = self._map_size
        world_xyz = voxel2pc(
            voxel_xyz, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size
        )
        return world_xyz

    def _points_to_voxel_map(self, points):
        """convert points in world frame to voxel frame, voxelize, and return the voxelized points"""
        _points = points.astype(np.float32)
        _voxels_bounds_robot_min = self._env.workspace_bounds_min.astype(np.float32)
        _voxels_bounds_robot_max = self._env.workspace_bounds_max.astype(np.float32)
        _map_size = self._map_size
        return pc2voxel_map(
            _points, _voxels_bounds_robot_min, _voxels_bounds_robot_max, _map_size
        )

    def _get_voxel_center(self, voxel_map):
        """calculte the center of the voxel map where value is 1"""
        voxel_center = np.array(np.where(voxel_map == 1)).mean(axis=1)
        return voxel_center

    def _get_scene_collision_voxel_map(self):
        collision_points_world, _ = self._env.get_scene_3d_obs(ignore_robot=True)
        collision_voxel = self._points_to_voxel_map(collision_points_world)
        return collision_voxel

    def _get_default_voxel_map(self, type="target"):
        """returns default voxel map (defaults to current state)"""

        def fn_wrapper():
            if type == "target":
                voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
            elif type == "obstacle":  # for LLM to do customization
                voxel_map = np.zeros((self._map_size, self._map_size, self._map_size))
            elif type == "velocity":
                voxel_map = np.ones((self._map_size, self._map_size, self._map_size))
            elif type == "gripper":
                voxel_map = (
                    np.ones((self._map_size, self._map_size, self._map_size))
                    * self._env.get_last_gripper_action()
                )
            elif type == "rotation":
                voxel_map = np.zeros(
                    (self._map_size, self._map_size, self._map_size, 4)
                )
                voxel_map[:, :, :] = self._env.get_ee_quat()
            else:
                raise ValueError("Unknown voxel map type: {}".format(type))
            voxel_map = VoxelIndexingWrapper(voxel_map)
            return voxel_map

        return fn_wrapper

    def _path2traj(self, path, rotation_map, velocity_map, gripper_map):
        """
        convert path (generated by planner) to trajectory (used by controller)
        path only contains a sequence of voxel coordinates, while trajectory parametrize the motion of the end-effector with rotation, velocity, and gripper on/off command
        """
        # convert path to trajectory
        traj = []
        for i in range(len(path)):
            # get the current voxel position
            voxel_xyz = path[i]
            # get the current world position
            world_xyz = self._voxel_to_world(voxel_xyz)
            voxel_xyz = np.round(voxel_xyz).astype(int)
            # get the current rotation (in world frame)
            rotation = (
                rotation_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
                if rotation_map is not None
                else None
            )
            # get the current velocity
            velocity = velocity_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
            # get the current on/off
            gripper = (
                gripper_map[voxel_xyz[0], voxel_xyz[1], voxel_xyz[2]]
                if gripper_map is not None
                else None
            )
            # LLM might specify a gripper value change, but sometimes EE may not be able to reach the exact voxel, so we overwrite the gripper value if it's close enough (TODO: better way to do this?)
            if (
                gripper is not None
                and (i == len(path) - 1)
                and not (np.all(gripper_map == 1) or np.all(gripper_map == 0))
            ):
                # get indices of the less common values
                less_common_value = (
                    1 if np.sum(gripper_map == 1) < np.sum(gripper_map == 0) else 0
                )
                less_common_indices = np.where(gripper_map == less_common_value)
                less_common_indices = np.array(less_common_indices).T
                # get closest distance from voxel_xyz to any of the indices that have less common value
                closest_distance = np.min(
                    np.linalg.norm(less_common_indices - voxel_xyz[None, :], axis=0)
                )
                # if the closest distance is less than threshold, then set gripper to less common value
                if closest_distance <= 3:
                    gripper = less_common_value
                    print(
                        f"{bcolors.OKBLUE}[interfaces.py | {get_clock_time()}] overwriting gripper to less common value for the last waypoint{bcolors.ENDC}"
                    )
            # add to trajectory
            traj.append((world_xyz, rotation, velocity, gripper))
        # append the last waypoint a few more times for the robot to stabilize
        for _ in range(2):
            traj.append((world_xyz, rotation, velocity, gripper))
        return traj

    def _preprocess_avoidance_map(self, avoidance_map, affordance_map, movable_obs):
        # # collision avoidance
        scene_collision_map = self._get_scene_collision_voxel_map()
        # anywhere within 10/100 indices of the target is ignored (to guarantee that we can reach the target)
        ignore_mask = distance_transform_edt(1 - affordance_map)
        scene_collision_map[ignore_mask < int(0.10 * self._map_size)] = 0
        # anywhere within 10/100 indices of the start is ignored, maybe can prevent the drone from being stuck
        try:
            # should arise KeyError if the object is the quadricopter
            ignore_mask = distance_transform_edt(1 - movable_obs["occupancy_map"])
            scene_collision_map[ignore_mask < int(0.10 * self._map_size)] = 0
        except KeyError:
            start_pos = movable_obs["position"]
            ignore_mask = np.ones_like(avoidance_map)
            ignore_mask[
                start_pos[0] - int(0.1 * self._map_size) : start_pos[0]
                + int(0.1 * self._map_size),
                start_pos[1] - int(0.1 * self._map_size) : start_pos[1]
                + int(0.1 * self._map_size),
                start_pos[2] - int(0.1 * self._map_size) : start_pos[2]
                + int(0.1 * self._map_size),
            ] = 0
            scene_collision_map *= ignore_mask
        avoidance_map += scene_collision_map
        avoidance_map = np.clip(avoidance_map, 0, 1)
        return avoidance_map


def setup_LMP(env, general_config, debug=False, engine_call_fn=None):
    controller_config = general_config["controller"]
    planner_config = general_config["planner"]
    lmp_env_config = general_config["lmp_config"]["env"]
    lmps_config = general_config["lmp_config"]["lmps"]
    env_name = general_config["env_name"]
    # LMP env wrapper
    lmp_env = LMP_interface(
        env, lmp_env_config, controller_config, planner_config, env_name=env_name
    )
    # creating APIs that the LMPs can interact with
    fixed_vars = {
        "np": np,
        "euler2quat": transforms3d.euler.euler2quat,
        "quat2euler": transforms3d.euler.quat2euler,
        "qinverse": transforms3d.quaternions.qinverse,
        "qmult": transforms3d.quaternions.qmult,
    }  # external library APIs
    variable_vars = {
        k: getattr(lmp_env, k)
        for k in dir(lmp_env)
        if callable(getattr(lmp_env, k)) and not k.startswith("_")
    }  # our custom APIs exposed to LMPs

    # allow LMPs to access other LMPs
    lmp_names = [
        name
        for name in lmps_config.keys()
        if not name in ["composer", "planner", "config"]
    ]
    low_level_lmps = {
        k: LMP(
            k,
            lmps_config[k],
            fixed_vars,
            variable_vars,
            debug,
            env_name,
            engine_call_fn=engine_call_fn,
        )
        for k in lmp_names
    }
    variable_vars.update(low_level_lmps)

    # creating the LMP for skill-level composition
    composer = LMP(
        "composer",
        lmps_config["composer"],
        fixed_vars,
        variable_vars,
        debug,
        env_name,
        engine_call_fn=engine_call_fn,
    )
    variable_vars["composer"] = composer

    # creating the LMP that deals w/ high-level language commands
    task_planner = LMP(
        "planner",
        lmps_config["planner"],
        fixed_vars,
        variable_vars,
        debug,
        env_name,
        engine_call_fn=engine_call_fn,
    )

    lmps = {
        "plan_ui": task_planner,
        "composer_ui": composer,
    }
    lmps.update(low_level_lmps)

    return lmps, lmp_env


# ======================================================
# jit-ready functions (for faster replanning time, need to install numba and add "@njit")
# ======================================================
def pc2voxel(pc, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
    """voxelize a point cloud"""
    pc = pc.astype(np.float32)
    # make sure the point is within the voxel bounds
    pc = np.clip(pc, voxel_bounds_robot_min, voxel_bounds_robot_max)
    # voxelize
    voxels = (
        (pc - voxel_bounds_robot_min)
        / (voxel_bounds_robot_max - voxel_bounds_robot_min)
        * (map_size - 1)
    )
    # to integer
    _out = np.empty_like(voxels)
    voxels = np.round(voxels, 0, _out).astype(np.int32)
    assert np.all(voxels >= 0), f"voxel min: {voxels.min()}"
    assert np.all(voxels < map_size), f"voxel max: {voxels.max()}"
    return voxels


def voxel2pc(voxels, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
    """de-voxelize a voxel"""
    # check voxel coordinates are non-negative
    assert np.all(voxels >= 0), f"voxel min: {voxels.min()}"
    assert np.all(voxels < map_size), f"voxel max: {voxels.max()}"
    voxels = voxels.astype(np.float32)
    # de-voxelize
    pc = (
        voxels / (map_size - 1) * (voxel_bounds_robot_max - voxel_bounds_robot_min)
        + voxel_bounds_robot_min
    )
    return pc


def pc2voxel_map(points, voxel_bounds_robot_min, voxel_bounds_robot_max, map_size):
    """given point cloud, create a fixed size voxel map, and fill in the voxels"""
    points = points.astype(np.float32)
    voxel_bounds_robot_min = voxel_bounds_robot_min.astype(np.float32)
    voxel_bounds_robot_max = voxel_bounds_robot_max.astype(np.float32)
    # make sure the point is within the voxel bounds
    points = np.clip(points, voxel_bounds_robot_min, voxel_bounds_robot_max)
    # voxelize
    voxel_xyz = (
        (points - voxel_bounds_robot_min)
        / (voxel_bounds_robot_max - voxel_bounds_robot_min)
        * (map_size - 1)
    )
    # to integer
    _out = np.empty_like(voxel_xyz)
    if np.nan or np.inf in voxel_xyz:
        dump_file = f"./dumps/voxel_xyz_{get_clock_time(True)}.txt"
        dump_file_replace = f"./dumps/voxel_xyz_replace_{get_clock_time(True)}.txt"
        np.savetxt(dump_file, voxel_xyz)
        print(
            f"nan or inf in voxel_xyz, please check {dump_file}, we will replace them with numbers, please check {dump_file_replace}"
        )
        voxel_xyz = np.nan_to_num(voxel_xyz, copy=True)
        np.savetxt(dump_file_replace, voxel_xyz)
    points_vox = np.round(voxel_xyz, 0, _out).astype(np.int32)
    voxel_map = np.zeros((map_size, map_size, map_size))
    for i in range(points_vox.shape[0]):
        voxel_map[points_vox[i, 0], points_vox[i, 1], points_vox[i, 2]] = 1
    return voxel_map
