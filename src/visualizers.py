"""Plotly-Based Visualizer"""
import plotly.graph_objects as go
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
        self.save_dir = config["save_dir"]
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.quality = config["quality"]
        self.update_quality(self.quality)
        self.map_size = config["map_size"]
        self.objects_points = []

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

    def update_scene_points(self, points, colors=None):
        points = points.astype(np.float16)
        if colors is None:
            colors = np.zeros((points.shape[0], 3), dtype=np.uint8) # default color is black
        assert colors.dtype == np.uint8
        self.scene_points = (points, colors)

    def add_object_points(self, points, label):
        points = points.astype(np.float16)
        self.objects_points.append((points, label))

    def update_depth_map(self, depth_map):
        self.depth_map = depth_map

    def update_mask(self, mask):
        self.mask = mask

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
                # Calculate the skip ratio for grid points
                skip_ratio = (self.workspace_bounds_max - self.workspace_bounds_min) / (
                    self.map_size / self.downsample_ratio
                )
                # Generate the grid points, mgrid is similar to meshgrid, which takes the start, end, and step size for each dimension
                x, y, z = np.mgrid[
                    self.workspace_bounds_min[0] : self.workspace_bounds_max[
                        0
                    ] : skip_ratio[0],
                    self.workspace_bounds_min[1] : self.workspace_bounds_max[
                        1
                    ] : skip_ratio[1],
                    self.workspace_bounds_min[2] : self.workspace_bounds_max[
                        2
                    ] : skip_ratio[2],
                ]
                grid_shape = costmap.shape
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
                        value=costmap.flatten(),
                        isomin=0,
                        isomax=1,
                        opacity=self.costmap_opacity,
                        surface_count=self.costmap_surface_count,
                        colorscale="Jet",
                        showlegend=True,
                        showscale=False,
                        name="costmap",
                    )
                )
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
                    marker=dict(size=3, color=scene_point_colors, opacity=1.0),
                )
            )
        # fig = go.Figure(data=fig_data)
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene", "rowspan": 2}, {"type": "scene"}],
                [
                    None,
                    {"type": "xy"},
                ],
            ],
            subplot_titles=[
                "display",
                "objects cloudpoints",
                None,
                "mask and depth map",
            ],
        )
        for trace in fig_data:
            fig.add_trace(trace, row=1, col=1)

        # set bounds and ratio
        fig.update_layout(
            scene=dict(
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
            ),
            scene_aspectmode="manual",
            scene_aspectratio=dict(
                x=self.scene_scale[0], y=self.scene_scale[1], z=self.scene_scale[2]
            ),
        )

        # do not show grid and axes
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=True, title="", visible=True),
                yaxis=dict(showgrid=False, showticklabels=True, title="", visible=True),
                zaxis=dict(showgrid=False, showticklabels=True, title="", visible=True),
            ),
        )

        # set background color as white
        fig.update_layout(template="none")

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
