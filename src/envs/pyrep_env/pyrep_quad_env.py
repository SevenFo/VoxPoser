import os, re, warnings
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import json

from pyrep.const import ObjectType, RenderMode
from pyrep.objects.vision_sensor import VisionSensor
from pyrep import PyRep

from utils import normalize_vector, bcolors, Observation
from visualizers import ValueMapVisualizer
from VLMPipline.VLM import VLM
from VLMPipline.utils import convert_depth_to_pointcloud
from envs.pyrep_env.pyrepqudacopter import PyRepQuadcopter


class CameraConfig(object):
    def __init__(
        self,
        rgb=True,
        depth=True,
        point_cloud=False,
        image_size=(128, 128),
        render_mode=RenderMode.OPENGL3,
        depth_in_meters=False,
    ):
        self.rgb = rgb
        self.depth = depth
        self.point_cloud = point_cloud
        self.image_size = image_size
        self.render_mode = render_mode
        self.depth_in_meters = depth_in_meters

    def set_all(self, value: bool):
        self.rgb = value
        self.depth = value
        self.point_cloud = value


def _set_rgb_props(rgb_cam: VisionSensor, rgb: bool, depth: bool, conf: CameraConfig):
    if not (rgb or depth or conf.point_cloud):
        rgb_cam.remove()
    else:
        rgb_cam.set_explicit_handling(1)
        rgb_cam.set_resolution(conf.image_size)
        rgb_cam.set_render_mode(conf.render_mode)


CAMERA_NAME_LIST = ["cam_NE", "cam_NW", "cam_SE", "cam_SW", "Quadricopter_cam"]


class VoxPoserPyRepQuadcopterEnv:
    def __init__(
        self,
        coppelia_scene_path="quadcopter.ttt",
        visualizer: ValueMapVisualizer = None,
        headless=False,
        vlmpipeline: VLM = None,
        target_objects=["quadcopter", "table"],
    ):
        """
        Initializes the VoxPoserPyRepQuadcopterEnv environment.

        Args:
            visualizer: Visualization interface, optional.
            headless (bool, optional): whether to run the environment in headless mode.
            VILMPipeline (optional): VILMPipeline object, input frame and objects output object masks.
        """
        self._target_objects = target_objects
        self._cam_config = CameraConfig(
            rgb=True,
            depth=True,
            point_cloud=False,
            image_size=(480, 480),
        )
        # create camera from camera name list
        self.camera_names = CAMERA_NAME_LIST
        self._cameras = {}
        self._pyrep = PyRep()
        self._pyrep.launch(coppelia_scene_path, headless=headless)

        for cam_name in CAMERA_NAME_LIST:
            # if camera not exist in PyRep, create it
            if not VisionSensor.exists(cam_name):
                raise ValueError(
                    f"Camera {cam_name} not found in the scene, we havent implement the function to create camera"
                )
            else:
                self._cameras[cam_name] = VisionSensor(cam_name)
        [
            _set_rgb_props(
                cam, self._cam_config.rgb, self._cam_config.depth, self._cam_config
            )
            for cam in self._cameras.values()
        ]  # set camera properties
        self.vlm = vlmpipeline

        self.workspace_bounds_min = np.array([-5, -5, 0])
        self.workspace_bounds_max = np.array([5, 5, 3])
        self.visualizer = visualizer
        if self.visualizer is not None:
            self.visualizer.update_bounds(
                self.workspace_bounds_min, self.workspace_bounds_max
            )

        forward_vector = np.array([0, 0, 1])
        self.lookat_vectors = {}
        self.camera_params = (
            {}
        )  # different camera has different params, key: camera name, value: params
        for cam_name in CAMERA_NAME_LIST:
            extrinsics = self._cameras[cam_name].get_matrix()
            lookat = extrinsics[:3, :3] @ forward_vector
            self.lookat_vectors[cam_name] = normalize_vector(lookat)
            self.camera_params.update(
                {
                    cam_name: {
                        "extrinsic_params": extrinsics,
                        "intrinsic_params": self._cameras[
                            cam_name
                        ].get_intrinsic_matrix(),
                        "far_near": (
                            self._cameras[cam_name].get_far_clipping_plane(),
                            self._cameras[cam_name].get_near_clipping_plane(),
                        ),
                    }
                }
            )
        # print(f"{bcolors.OKGREEN}Camera parameters:{bcolors.ENDC} {self.camera_params}")

        # get quadcopter object
        self.quadcopter = PyRepQuadcopter()
        self.init_task()
        self.has_processed_first_frame = False

    def get_object_names(self):
        """
        Returns the names of all objects in the current task environment.

        Returns:
            list: A list of object names.
        """
        return self._target_objects

    def init_task(self):
        """
        Loads a new task into the environment and resets task-related variables.
        Records the mask IDs of the robot, gripper, and objects in the scene.

        Args:
            task (str or rlbench.tasks.Task): Name of the task class or a task object.
        """
        self.target_objects = list(
            set(re.sub(r"\d+", "", name) for name in self.get_object_names())
        )
        print(f"we only process these objects: {self.target_objects}")
        # set max instances for all object for category_multiplier
        self.category_multiplier = 100  # which means the instance number of each object categery is less than 100
        # it should be attention the category label is different from the object label
        # object label = (category label + 1) * category_multiplier + instance id as 0 represent background
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
        self.descriptions = [
            "fly around a distance above the table",
            "From under the table, cross the past to the 100cm in front of the table, then fly to the top 100cm above the table",
            "fly to the table",
            "go to the table",
        ]

        self._pyrep.start()

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
            depth_frame = getattr(self.latest_obs, f"{cam}_depth")
            mask_frame = self.latest_mask[cam]
            point = convert_depth_to_pointcloud(
                depth_image=depth_frame,
                extrinsic_params=self.camera_params[cam]["extrinsic_params"],
                camera_intrinsics=self.camera_params[cam]["intrinsic_params"],
                clip_far=self.camera_params[cam]["far_near"][0],
                clip_near=self.camera_params[cam]["far_near"][1],
            )
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
        points, masks, normals = [], [], []
        for cam in self.camera_names:
            depth_frame = getattr(self.latest_obs, f"{cam}_depth")
            point = convert_depth_to_pointcloud(
                depth_image=depth_frame,
                extrinsic_params=self.camera_params[cam]["extrinsic_params"],
                camera_intrinsics=self.camera_params[cam]["intrinsic_params"],
                clip_far=self.camera_params[cam]["far_near"][0],
                clip_near=self.camera_params[cam]["far_near"][1],
            )
            points.append(point.reshape(-1, 3))
        points = np.concatenate(points, axis=0)
        return points, None

    def get_obs(self):
        """get observation from the environment, including vision sensor data and the current position of quadcopter
        TODO: add noise to the observation
        TODO: get observation from other sensor
        """

        def get_rgb_depth(
            sensor: VisionSensor,
            get_rgb: bool,
            get_depth: bool,
            get_pcd: bool,
            rgb_noise=None,
            depth_noise=None,
            depth_in_meters: bool = False,
        ):
            rgb = depth = pcd = None
            if sensor is not None and (get_rgb or get_depth):
                sensor.handle_explicitly()
                if get_rgb:
                    rgb = sensor.capture_rgb()
                    if rgb_noise is not None:
                        rgb = rgb_noise.apply(rgb)
                    rgb = np.clip((rgb * 255.0).astype(np.uint8), 0, 255)
                if get_depth or get_pcd:
                    depth = sensor.capture_depth(depth_in_meters)
                    if depth_noise is not None:
                        depth = depth_noise.apply(depth)
                if get_pcd:
                    depth_m = depth
                    if not depth_in_meters:
                        near = sensor.get_near_clipping_plane()
                        far = sensor.get_far_clipping_plane()
                        depth_m = near + depth * (far - near)
                    pcd = sensor.pointcloud_from_depth(depth_m)
                    if not get_depth:
                        depth = None
            return rgb, depth, pcd

        obs = {}
        for cam in self.camera_names:
            rgb, depth, pcd = get_rgb_depth(
                self._cameras[cam],
                self._cam_config.rgb,
                self._cam_config.depth,
                self._cam_config.point_cloud,
                rgb_noise=None,
                depth_noise=None,
                depth_in_meters=self._cam_config.depth_in_meters,
            )
            obs.update({f"{cam}_rgb": rgb, f"{cam}_depth": depth, f"{cam}_pcd": pcd})

        # set quadcopter position
        obs.update(
            {
                "quad_pose": np.concatenate(
                    [
                        self.quadcopter.get_position(),
                        self.quadcopter.get_quaternion(),
                        self.quadcopter.get_orientation(),
                    ]
                )
            }
        )

        return self._process_obs(
            Observation(obs)
        )  # convet dict to Observation object, which can access by obs.key

    def reset(self):
        """
        Resets the environment and the task. Also updates the visualizer.

        Returns:
            tuple: A tuple containing task descriptions and initial observations.
        """
        # todo
        self._pyrep.stop()
        self._pyrep.start()

        descriptions = self.descriptions[0]
        obs = self.get_obs()
        self.init_obs = obs
        self.latest_obs = obs
        self.latest_mask = {}
        rgb_frames = {}  # in c w h
        self.target_objects_labels = list(range(len(self.target_objects)))
        for cam in self.camera_names:
            rgb_frames[cam] = getattr(self.latest_obs, f"{cam}_rgb").transpose(
                [2, 0, 1]
            )
        frames = np.stack(list(rgb_frames.values()), axis=0)
        masks = self.vlm.process_first_frame(
            self.target_objects, frames, verbose=True, owlv2_threshold=0.1
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
        return descriptions, obs

    def apply_action(self, action):
        """
        Applies an action in the environment and updates the state
        while the action is the target position of qudacopter in form of [x,y,z,quaternion], where quaternion is in form of [x,y,z,w], orientation is in form of [roll,pitch,yaw]
        however, we only apply the target position (x,y,z), the orientation has not been implemented yet, which means the orientation of quadcopter is fixed, or keep horizontal
        attention: the process would be blocked until the quadcopter reach the target position

        Args:
            action: The action to apply.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        assert (
            self.has_processed_first_frame
        ), "Please reset the environment first or let VLM process the first frame"
        assert (
            len(action) == 7
        ), "the action should be [x,y,z,quaternion] where quaternion is in form of [x,y,z,w]"
        action = self._process_action(action)
        # actually before apply action, we should check whether the action is valid, while we haven't implement it
        remainder = self.quadcopter.goto(action[:3])
        # block until the quadcopter reach the target position
        if len(remainder) == 0:
            while not self.quadcopter.action_is_reached(
                action[:3]
            ) or self.quadcopter.action_is_failed(action[:3]):
                self._pyrep.step()  # step the simulation
        else:
            remainder = iter(remainder)
            current_target = next(remainder)
            while True:
                while not self.quadcopter.action_is_reached(
                    current_target
                ) or self.quadcopter.action_is_failed(current_target):
                    self._pyrep.step()
                try:
                    print("go to next step remainded")
                    current_target = next(remainder)
                except StopIteration as e:
                    self.quadcopter.goto(action[:3], ignor_distance=True)
                    while not self.quadcopter.action_is_reached(
                        action[:3]
                    ) or self.quadcopter.action_is_failed(action[:3]):
                        self._pyrep.step()
                    assert self.quadcopter.action_is_reached(
                        action[:3]
                    ), "something worong happed as the action havent reached"
                    break
                self.quadcopter.goto(current_target, ignor_distance=True)
        print("action is finished")
        obs = self.get_obs()
        reward = terminate = None
        self.latest_obs = obs
        self.latest_reward = reward  # TODO
        self.latest_terminate = terminate  # TODO
        self.latest_action = action
        # use VLM to get the mask of the intrerested objects from rgb image from latest_obs
        rgb_frames = {}  # in c w h
        self.target_objects_labels = list(range(len(self.target_objects)))
        for cam in self.camera_names:
            rgb_frames[cam] = getattr(self.latest_obs, f"{cam}_rgb").transpose(
                [2, 0, 1]
            )
        frames = np.stack(list(rgb_frames.values()), axis=0)
        masks = self.vlm.process_frame(frames, verbose=True)
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
        return obs, reward, terminate

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        init_pose = self.init_obs.quad_pose
        return self.apply_action(init_pose)

    def get_ee_pose(self):
        assert self.latest_obs is not None, "Please reset the environment first"
        return self.latest_obs.quad_pose

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[3:7]

    def get_ee_orientation(self):
        return self.get_ee_pose()[7:]

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

    def _update_visualizer(self):
        """
        Updates the scene in the visualizer with the latest observations.
        Update after each step. (reset and applid action)

        Note: This function is generally called internally.
        """
        if self.visualizer is not None:
            points, colors = self.get_scene_3d_obs(
                ignore_robot=False, ignore_grasped_obj=False
            )
            self.visualizer.update_scene_points(points, colors)
            fig = plt.figure(figsize=(6.4 * len(self.camera_names), 4.8))
            for idx, cam in enumerate(self.camera_names):
                rgb = getattr(self.latest_obs, f"{cam}_rgb")
                mask = np.mod(self.latest_mask[cam], 256)  # avoid overflow for color
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

    def _process_obs(self, obs):
        """
        Processes the observations, specifically converts quaternion format from xyzw to wxyz.

        Args:
            obs: The observation to process.

        Returns:
            The processed observation.
        """
        return obs

    def _process_action(self, action):
        """
        Processes the action, specifically converts quaternion format from wxyz to xyzw.

        Args:
            action: The action to process.

        Returns:
            The processed action.
        """
        return action

    def _set_camera_properties(self) -> None:
        def _set_rgb_props(
            rgb_cam: VisionSensor, rgb: bool, depth: bool, conf: CameraConfig
        ):
            if not (rgb or depth or conf.point_cloud):
                rgb_cam.remove()
            else:
                rgb_cam.set_explicit_handling(1)
                rgb_cam.set_resolution(conf.image_size)
                rgb_cam.set_render_mode(conf.render_mode)

        def _set_mask_props(mask_cam: VisionSensor, mask: bool, conf: CameraConfig):
            if not mask:
                mask_cam.remove()
            else:
                mask_cam.set_explicit_handling(1)
                mask_cam.set_resolution(conf.image_size)

        _set_rgb_props(
            self._cam_over_shoulder_left,
            self._obs_config.left_shoulder_camera.rgb,
            self._obs_config.left_shoulder_camera.depth,
            self._obs_config.left_shoulder_camera,
        )
        _set_rgb_props(
            self._cam_over_shoulder_right,
            self._obs_config.right_shoulder_camera.rgb,
            self._obs_config.right_shoulder_camera.depth,
            self._obs_config.right_shoulder_camera,
        )
        _set_rgb_props(
            self._cam_overhead,
            self._obs_config.overhead_camera.rgb,
            self._obs_config.overhead_camera.depth,
            self._obs_config.overhead_camera,
        )
        _set_rgb_props(
            self._cam_wrist,
            self._obs_config.wrist_camera.rgb,
            self._obs_config.wrist_camera.depth,
            self._obs_config.wrist_camera,
        )
        _set_rgb_props(
            self._cam_front,
            self._obs_config.front_camera.rgb,
            self._obs_config.front_camera.depth,
            self._obs_config.front_camera,
        )
        _set_mask_props(
            self._cam_over_shoulder_left_mask,
            self._obs_config.left_shoulder_camera.mask,
            self._obs_config.left_shoulder_camera,
        )
        _set_mask_props(
            self._cam_over_shoulder_right_mask,
            self._obs_config.right_shoulder_camera.mask,
            self._obs_config.right_shoulder_camera,
        )
        _set_mask_props(
            self._cam_overhead_mask,
            self._obs_config.overhead_camera.mask,
            self._obs_config.overhead_camera,
        )
        _set_mask_props(
            self._cam_wrist_mask,
            self._obs_config.wrist_camera.mask,
            self._obs_config.wrist_camera,
        )
        _set_mask_props(
            self._cam_front_mask,
            self._obs_config.front_camera.mask,
            self._obs_config.front_camera,
        )
