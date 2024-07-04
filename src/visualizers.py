"""Plotly-Based Visualizer"""

import plotly.graph_objects as go
import imageio.v3 as iio
from plotly.subplots import make_subplots
import numpy as np
import os
import datetime


class ValueMapVisualizer:
    """
    A Plotly-based visualizer for 3D value map and planned path.
    """

    def __init__(self, config):
        self.scene_points = None
        self.scene_points_filted = None
        self.save_dir = config["save_dir"]
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.quality = config["quality"]
        self.update_quality(self.quality)
        self.map_size = config["map_size"]
        self.objects_points = []
        self.masks = {}
        self.rgb = {}
        self.frames = []

    def update_bounds(self, lower, upper):
        self.workspace_bounds_min = lower
        self.workspace_bounds_max = upper
        self.plot_bounds_min = lower - 0.15 * (upper - lower)
        self.plot_bounds_max = upper + 0.15 * (upper - lower)
        xyz_ratio = 1 / (self.workspace_bounds_max - self.workspace_bounds_min)
        scene_scale = np.max(xyz_ratio) / xyz_ratio
        self.scene_scale = scene_scale

    def update_quality(self, quality):
        self.quality = quality
        if self.quality == "low":
            self.downsample_ratio = 4
            self.max_scene_points = 150000
            self.costmap_opacity = 0.2 * 0.6
            self.costmap_surface_count = 10
        elif self.quality == "low-full-scene":
            self.downsample_ratio = 4
            self.max_scene_points = 1000000
            self.costmap_opacity = 0.2 * 0.6
            self.costmap_surface_count = 10
        elif self.quality == "low-half-scene":
            self.downsample_ratio = 4
            self.max_scene_points = 250000
            self.costmap_opacity = 0.2 * 0.6
            self.costmap_surface_count = 10
        elif self.quality == "medium":
            self.downsample_ratio = 2
            self.max_scene_points = 300000
            self.costmap_opacity = 0.1 * 0.6
            self.costmap_surface_count = 30
        elif self.quality == "medium-full-scene":
            self.downsample_ratio = 2
            self.max_scene_points = 1000000
            self.costmap_opacity = 0.1 * 0.6
            self.costmap_surface_count = 30
        elif self.quality == "medium-half-scene":
            self.downsample_ratio = 2
            self.max_scene_points = 500000
            self.costmap_opacity = 0.1 * 0.6
            self.costmap_surface_count = 30
        elif self.quality == "high":
            self.downsample_ratio = 1
            self.max_scene_points = 500000
            self.costmap_opacity = 0.07 * 0.6
            self.costmap_surface_count = 50
        elif self.quality == "best":
            self.downsample_ratio = 1
            self.max_scene_points = 500000
            self.costmap_opacity = 0.05 * 0.6
            self.costmap_surface_count = 100
        else:
            raise ValueError(
                f"Unknown quality: {self.quality}; should be one of [low, medium, high]"
            )

    def update_scene_points(self, points: np.array, colors=None):
        minx, miny, minz = self.workspace_bounds_min
        maxx, maxy, maxz = self.workspace_bounds_max
        mask = np.stack(
            [
                points[:, 0] < minx + 0.1,
                points[:, 0] > maxx - 0.1,
                points[:, 1] < miny + 0.1,
                points[:, 1] > maxy - 0.1,
                points[:, 2] < minz + 0.1,
                points[:, 2] > maxz - 0.1,
            ],
            axis=1,
        )
        mask = np.logical_not(np.any(mask, axis=1))
        points = points[mask, :]
        points = points.astype(np.float16)
        self.scene_points = (points, colors)

    def update_scene_points_filted(self, points: np.array, colors=None):
        minx, miny, minz = self.workspace_bounds_min
        maxx, maxy, maxz = self.workspace_bounds_max
        mask = np.stack(
            [
                points[:, 0] < minx + 0.1,
                points[:, 0] > maxx - 0.1,
                points[:, 1] < miny + 0.1,
                points[:, 1] > maxy - 0.1,
                points[:, 2] < minz + 0.1,
                points[:, 2] > maxz - 0.1,
            ],
            axis=1,
        )
        mask = np.logical_not(np.any(mask, axis=1))
        points = points[mask, :]
        points = points.astype(np.float16)
        if colors is None:
            colors = np.zeros(
                (points.shape[0], 3), dtype=np.uint8
            )  # default color is black
        assert colors.dtype == np.uint8
        self.scene_points_filted = (points, colors)

    def add_object_points(self, points, label):
        added_labels = [l for _, l in self.objects_points]
        if label not in added_labels:
            points = points.astype(np.float16)
            self.objects_points.append((points, label))

    def update_depth_map(self, depth_map):
        self.depth_map = depth_map

    def add_mask(self, cam_name, mask):
        if cam_name not in self.masks.keys():
            self.masks.update({f"{cam_name}": [mask]})
            return
        self.masks[cam_name].append(mask)

    def add_rgb(self, cam_name, rgb):
        if cam_name not in self.rgb.keys():
            self.rgb.update({f"{cam_name}": [rgb]})
            return
        self.rgb[cam_name].append(rgb)

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_gifs(self):
        curr_time = datetime.datetime.now()
        log_id = f"{curr_time.hour}:{curr_time.minute}:{curr_time.second}"
        if len(self.frames) > 0:
            save_path = os.path.join(self.save_dir, log_id + "-frames.gif")
            print(f"** saving frames gif to {save_path}")
            iio.imwrite(save_path, np.stack(self.frames, axis=0), fps=10)
        if len(self.masks) > 0:
            for key, value in self.masks.items():
                save_path = os.path.join(self.save_dir, log_id + f"-{key}-masks.gif")
                print(f"** saving masks gif {key} to {save_path}")
                iio.imwrite(save_path, np.stack(value, axis=0), fps=10)
        if len(self.rgb) > 0:
            for key, value in self.rgb.items():
                save_path = os.path.join(self.save_dir, log_id + f"-{key}-rgb.gif")
                print(f"** saving rgb gif {key} to {save_path}")
                iio.imwrite(save_path, np.stack(value, axis=0), fps=10)
        print("** saved to", self.save_dir)
        self.masks = {}
        self.frames = []
        self.rgb = {}

    def _add_voxel_map(self, map, name, fig_data):
        skip_ratio = (self.workspace_bounds_max - self.workspace_bounds_min) / (
            self.map_size / self.downsample_ratio
        )
        # Generate the grid points, mgrid is similar to meshgrid, which takes the start, end, and step size for each dimension
        x, y, z = np.mgrid[
            self.workspace_bounds_min[0] : self.workspace_bounds_max[0] : skip_ratio[0],
            self.workspace_bounds_min[1] : self.workspace_bounds_max[1] : skip_ratio[1],
            self.workspace_bounds_min[2] : self.workspace_bounds_max[2] : skip_ratio[2],
        ]
        grid_shape = map.shape
        # Trim the grid points to match the costmap shape
        x = x[: grid_shape[0], : grid_shape[1], : grid_shape[2]]
        y = y[: grid_shape[0], : grid_shape[1], : grid_shape[2]]
        z = z[: grid_shape[0], : grid_shape[1], : grid_shape[2]]
        # Add the costmap as a volume plot
        fig_data.append(
            go.Volume(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value=map.flatten(),
                isomin=0,
                isomax=1,
                opacity=self.costmap_opacity,
                surface_count=self.costmap_surface_count,
                colorscale="Jet",
                showlegend=True,
                showscale=False,
                name=name,
            )
        )

    def visualize(self, info, show=False, save=True):
        """
        Visualize the path and relevant info using plotly.

        Args:
            info (dict): Dictionary containing the relevant information for visualization.
            show (bool, optional): Whether to display the visualization. Defaults to False.
            save (bool, optional): Whether to save the visualization. Defaults to True.

        Returns:
            go.Figure: The plotly figure object representing the visualization.
        """
        planner_info = info["planner_info"]
        waypoints_world = np.array([p[0] for p in info["traj_world"]])
        start_pos_world = info["start_pos_world"]
        assert len(start_pos_world.shape) == 1
        waypoints_world = np.concatenate(
            [start_pos_world[None, ...], waypoints_world], axis=0
        )
        fig_data = []
        # plot path
        # add marker to path waypoints
        fig_data.append(
            go.Scatter3d(
                x=waypoints_world[:, 0],
                y=waypoints_world[:, 1],
                z=waypoints_world[:, 2],
                mode="markers",
                name="waypoints",
                marker=dict(size=4, color="red"),
            )
        )
        # add lines between waypoints
        for i in range(waypoints_world.shape[0] - 1):
            fig_data.append(
                go.Scatter3d(
                    x=waypoints_world[i : i + 2, 0],
                    y=waypoints_world[i : i + 2, 1],
                    z=waypoints_world[i : i + 2, 2],
                    mode="lines",
                    name="path",
                    line=dict(width=10, color="orange"),
                )
            )
        if planner_info is not None:
            # plot costmap costmap = (target_map * self.config.target_map_weight+ obstacle_map * self.config.obstacle_map_weight)
            if "costmap" in planner_info:
                # Downsample the costmap, :: means sample points every downsample_ratio
                costmap = planner_info["costmap"][
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                ]
                self._add_voxel_map(costmap, "costmap", fig_data)
            if "affordance_map" in info:
                # Downsample the costmap, :: means sample points every downsample_ratio
                affordance_map = info["affordance_map"][
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                ]
                self._add_voxel_map(affordance_map, "affordance_map", fig_data)
            if "avoidance_map" in info:
                # Downsample the costmap, :: means sample points every downsample_ratio
                avoidance_map = info["avoidance_map"][
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                ]
                self._add_voxel_map(avoidance_map, "avoidance_map", fig_data)
            if "pre_avoidance_map" in info:
                # Downsample the costmap, :: means sample points every downsample_ratio
                pre_avoidance_map = info["pre_avoidance_map"][
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                ]
                self._add_voxel_map(pre_avoidance_map, "pre_avoidance_map", fig_data)
            # plot start position
            if "start_pos" in planner_info:
                fig_data.append(
                    go.Scatter3d(
                        x=[start_pos_world[0]],
                        y=[start_pos_world[1]],
                        z=[start_pos_world[2]],
                        mode="markers",
                        name="start",
                        marker=dict(size=6, color="blue"),
                    )
                )
            # plot target as dots extracted from target_map
            if "raw_target_map" in planner_info:
                targets_world = info["targets_world"]
                fig_data.append(
                    go.Scatter3d(
                        x=targets_world[:, 0],
                        y=targets_world[:, 1],
                        z=targets_world[:, 2],
                        mode="markers",
                        name="target",
                        marker=dict(size=6, color="green", opacity=0.7),
                    )
                )

        # visualize scene points
        if self.scene_points is None:
            print("no scene points to overlay, skipping...")
            scene_points = None
        else:
            scene_points, scene_point_colors = self.scene_points
            # resample to reduce the number of points
            if scene_points.shape[0] > self.max_scene_points:
                resample_idx = np.random.choice(
                    scene_points.shape[0],
                    min(scene_points.shape[0], self.max_scene_points),
                    replace=False,
                )
                scene_points = scene_points[resample_idx]
                if scene_point_colors is not None:
                    scene_point_colors = scene_point_colors[resample_idx]
            if scene_point_colors is None:
                scene_point_colors = scene_points[:, 2]
            else:
                scene_point_colors = scene_point_colors / 255.0
            # add scene points
            fig_data.append(
                go.Scatter3d(
                    x=scene_points[:, 0],
                    y=scene_points[:, 1],
                    z=scene_points[:, 2],
                    mode="markers",
                    name="scene points",
                    marker=dict(size=3, color=scene_point_colors, opacity=1.0),
                )
            )

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene", "rowspan": 1}, {"type": "scene"}],
                [
                    {"type": "scene"},
                    {"type": "xy"},
                ],
            ],
            subplot_titles=[
                "display",
                "objects cloudpoints",
                "scene_points_filtered",
                "mask and depth map",
            ],
        )
        for trace in fig_data:
            fig.add_trace(trace, row=1, col=1)

        # set bounds and ratio
        fig.update_scenes(
            xaxis=dict(
                range=[self.plot_bounds_min[0], self.plot_bounds_max[0]],
                autorange=False,
            ),
            yaxis=dict(
                range=[self.plot_bounds_min[1], self.plot_bounds_max[1]],
                autorange=False,
            ),
            zaxis=dict(
                range=[self.plot_bounds_min[2], self.plot_bounds_max[2]],
                autorange=False,
            ),
            aspectmode="manual",
            aspectratio=dict(
                x=self.scene_scale[0], y=self.scene_scale[1], z=self.scene_scale[2]
            ),
        )

        # do not show grid and axes
        fig.update_scenes(
            xaxis=dict(showgrid=False, showticklabels=True, title="", visible=True),
            yaxis=dict(showgrid=False, showticklabels=True, title="", visible=True),
            zaxis=dict(showgrid=False, showticklabels=True, title="", visible=True),
        )

        # add scene_points_filted to the second subplot
        # visualize scene points
        if self.scene_points_filted is None:
            print("no scene points to overlay, skipping...")
            scene_points = None
        else:
            scene_points, scene_point_colors = self.scene_points_filted
            # resample to reduce the number of points
            if scene_points.shape[0] > self.max_scene_points:
                resample_idx = np.random.choice(
                    scene_points.shape[0],
                    min(scene_points.shape[0], self.max_scene_points),
                    replace=False,
                )
                scene_points = scene_points[resample_idx]
                if scene_point_colors is not None:
                    scene_point_colors = scene_point_colors[resample_idx]
            if scene_point_colors is None:
                scene_point_colors = scene_points[:, 2]
            else:
                scene_point_colors = scene_point_colors / 255.0
            # add scene points
            fig.add_trace(
                go.Scatter3d(
                    x=scene_points[:, 0],
                    y=scene_points[:, 1],
                    z=scene_points[:, 2],
                    mode="markers",
                    name="scene points filtered",
                    marker=dict(size=3, color=scene_point_colors, opacity=1.0),
                ),
                row=2,
                col=1,
            )

        # set background color as white
        # fig.update_scenes(template="none")
        fig.add_trace(
            go.Scatter(
                x=[1, 2, 3],
                y=[4, 5, 6],
                mode="markers",
                name="mask figure",
                marker=dict(size=4, color="green"),
            ),
            row=2,
            col=2,
        )

        # add objects_points
        for obj_points, label in self.objects_points:
            # resample to reduce the number of points
            if obj_points.shape[0] > self.max_scene_points:
                resample_idx = np.random.choice(
                    obj_points.shape[0],
                    min(obj_points.shape[0], self.max_scene_points),
                    replace=False,
                )
                obj_points = obj_points[resample_idx]
            # add scene points
            fig.add_trace(
                go.Scatter3d(
                    x=obj_points[:, 0],
                    y=obj_points[:, 1],
                    z=obj_points[:, 2],
                    mode="markers",
                    name=label,
                    marker=dict(size=3, color="blue", opacity=1.0),
                ),
                row=1,
                col=2,
            )
        self.objects_points = []

        # save and show
        if save and self.save_dir is not None:
            curr_time = datetime.datetime.now()
            log_id = f"{curr_time.hour}:{curr_time.minute}:{curr_time.second}"
            save_path = os.path.join(self.save_dir, log_id + ".html")
            latest_save_path = os.path.join(self.save_dir, "latest.html")
            print("** saving visualization to", save_path, "...")
            fig.write_html(save_path)
            print("** saving visualization to", latest_save_path, "...")
            fig.write_html(latest_save_path)
            print(f"** save to {save_path}")
        if show:
            fig.show()

        return fig

    def visualize_plan_result(self, info):
        planner_info = info["planner_info"]
        waypoints_world = np.array([p[0] for p in info["traj_world"]])
        start_pos_world = info["start_pos_world"].squeeze()
        target_world = info["target_world"].squeeze()
        plan_iter = info["plan_iter"]
        assert len(start_pos_world.shape) == 1
        waypoints_world = np.concatenate(
            [start_pos_world[None, ...], waypoints_world], axis=0
        )
        fig_data = []
        # plot path
        # add marker to path waypoints
        fig_data.append(
            go.Scatter3d(
                x=waypoints_world[:, 0],
                y=waypoints_world[:, 1],
                z=waypoints_world[:, 2],
                mode="markers",
                name="waypoints",
                marker=dict(size=4, color="red"),
            )
        )
        # add lines between waypoints
        for i in range(waypoints_world.shape[0] - 1):
            fig_data.append(
                go.Scatter3d(
                    x=waypoints_world[i : i + 2, 0],
                    y=waypoints_world[i : i + 2, 1],
                    z=waypoints_world[i : i + 2, 2],
                    mode="lines",
                    name="path",
                    line=dict(width=10, color="orange"),
                )
            )
        # add start pos marker
        fig_data.append(
            go.Scatter3d(
                x=[start_pos_world[0]],
                y=[start_pos_world[1]],
                z=[start_pos_world[2]],
                mode="markers",
                name="start",
                marker=dict(size=6, color="blue"),
            )
        )
        # add target point marker
        fig_data.append(
            go.Scatter3d(
                x=[target_world[0]],
                y=[target_world[1]],
                z=[target_world[2]],
                mode="markers",
                name="target",
                marker=dict(size=6, color="green"),
            )
        )
        if planner_info is not None:
            if "costmap" in planner_info:
                # Downsample the costmap, :: means sample points every downsample_ratio
                costmap = planner_info["costmap"][
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                ]
                self._add_voxel_map(costmap, "costmap", fig_data)

            if "target_map" in planner_info:
                target_map = planner_info["target_map"][
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                ]
                self._add_voxel_map(target_map, "target_map", fig_data)

            if "obstacle_map" in planner_info:
                obstacle_map = planner_info["obstacle_map"][
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                    :: self.downsample_ratio,
                ]
                self._add_voxel_map(obstacle_map, "obstacle_map", fig_data)

        # scence points
        if self.scene_points is None:
            print("no scene points to overlay, skipping...")
            scene_points = None
        else:
            scene_points, scene_point_colors = self.scene_points
            # resample to reduce the number of points
            if scene_points.shape[0] > self.max_scene_points:
                resample_idx = np.random.choice(
                    scene_points.shape[0],
                    min(scene_points.shape[0], self.max_scene_points),
                    replace=False,
                )
                scene_points = scene_points[resample_idx]
                if scene_point_colors is not None:
                    scene_point_colors = scene_point_colors[resample_idx]
            if scene_point_colors is None:
                scene_point_colors = scene_points[:, 2]  # use depth as color
            else:
                scene_point_colors = scene_point_colors / 255.0
            # add scene points
            fig_data.append(
                go.Scatter3d(
                    x=scene_points[:, 0],
                    y=scene_points[:, 1],
                    z=scene_points[:, 2],
                    mode="markers",
                    name="scene points",
                    marker=dict(size=1, color=scene_point_colors, opacity=1.0),
                )
            )

        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"type": "scene"}]],
            subplot_titles=["display"],
        )

        for trace in fig_data:
            fig.add_trace(trace, row=1, col=1)

        # set bounds and ratio
        fig.update_scenes(
            xaxis=dict(
                range=[self.plot_bounds_min[0], self.plot_bounds_max[0]],
                autorange=False,
            ),
            yaxis=dict(
                range=[self.plot_bounds_min[1], self.plot_bounds_max[1]],
                autorange=False,
            ),
            zaxis=dict(
                range=[self.plot_bounds_min[2], self.plot_bounds_max[2]],
                autorange=False,
            ),
            aspectmode="manual",
            aspectratio=dict(
                x=self.scene_scale[0], y=self.scene_scale[1], z=self.scene_scale[2]
            ),
        )
        # do not show grid and axes
        fig.update_scenes(
            xaxis=dict(showgrid=False, showticklabels=True, title="", visible=True),
            yaxis=dict(showgrid=False, showticklabels=True, title="", visible=True),
            zaxis=dict(showgrid=False, showticklabels=True, title="", visible=True),
        )
        if self.save_dir is not None:
            # TODO concurrent.futures
            time = datetime.datetime.now()
            log_id = f"{time.hour}-{time.minute}-{time.second}_plan_iter_" + str(
                plan_iter
            )
            save_path = os.path.join(self.save_dir, log_id + ".html")
            latest_save_path = os.path.join(self.save_dir, "latest.html")
            print("** saving visualization to", save_path, "...")
            fig.write_html(save_path)
            print("** saving visualization to", latest_save_path, "...")
            fig.write_html(latest_save_path)
            print(f"** save to {save_path}")
