#!/usr/bin/python3
import rospy
import cv_bridge
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo, Image
from std_msgs.msg import Header
from geometry_msgs.msg import Transform, TransformStamped
import numpy as np
import cv2
from VLMPipline.VLM import VLM
import torch
import open3d as o3d
import tf2_ros, transforms3d
from tf2_msgs.msg import TFMessage
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import threading

import time


class ConfigurableTransformListener(tf2_ros.TransformListener):
    def __init__(
        self,
        buffer,
        queue_size=None,
        buff_size=65536,
        tcp_nodelay=False,
        tf_topic="tf",
        tf_static_topic="tf_static",
    ):
        # Copy past the code from the tf2_ros.TransformListener constructor method, changing the subscribed topics.
        self.buffer = buffer
        self.last_update = rospy.Time.now()
        self.last_update_lock = threading.Lock()
        self.tf_sub = rospy.Subscriber(
            tf_topic,
            TFMessage,
            self.callback,
            queue_size=queue_size,
            buff_size=buff_size,
            tcp_nodelay=tcp_nodelay,
        )
        self.tf_static_sub = rospy.Subscriber(
            tf_static_topic,
            TFMessage,
            self.static_callback,
            queue_size=queue_size,
            buff_size=buff_size,
            tcp_nodelay=tcp_nodelay,
        )


class DepthToPointCloudNode:
    def __init__(self, vlmpipeline: VLM = None):
        rospy.init_node("depth_to_pointcloud_node")
        self.world_frame_id = "World/World"
        self.rgb_frame_id = "World/EN"
        self.depth_frame_id = self.rgb_frame_id
        self._cvb = cv_bridge.CvBridge()
        rospy.loginfo("wait for tf message")

        self.camera_info_sub_depth = rospy.Subscriber(
            "/World/EC/camera_info",
            CameraInfo,
            lambda msg: [
                setattr(self, "intrinsic_matrix_depth", np.array(msg.K).reshape(3, 3)),
                setattr(self, "image_size", (msg.height, msg.width)),
            ],
        )
        self.camera_info_sub_rgb = rospy.Subscriber(
            "/World/EC/camera_info",
            CameraInfo,
            lambda msg: [
                setattr(self, "intrinsic_matrix_rgb", np.array(msg.K).reshape(3, 3)),
                setattr(self, "image_size_rgb", (msg.height, msg.width)),
            ],
        )
        # self.tf_world2depth_sub = rospy.Subscriber("/device_0/sensor_0/Depth_0/tf/0", Transform, lambda msg: setattr(self, "world2depth", msg))
        # world_frame2rgb_frame = rospy.wait_for_message(
        #     "/device_0/sensor_1/Color_0/tf/0", Transform
        # )
        while not hasattr(self, "image_size") or not hasattr(self, "image_size_rgb"):
            rospy.sleep(1)
        assert (
            self.image_size == self.image_size_rgb
        ), "depth and rgb image size should be the same"

        # static_tf_boardcaster = tf2_ros.StaticTransformBroadcaster()
        # world_frame2depth_frame = self.world2depth
        # world_frame2depth_frame_trans = TransformStamped()
        # world_frame2depth_frame_trans.header.frame_id = self.world_frame_id
        # world_frame2depth_frame_trans.header.stamp = rospy.Time.now()
        # world_frame2depth_frame_trans.child_frame_id = self.depth_frame_id
        # world_frame2depth_frame_trans.transform = world_frame2depth_frame
        # world_frame2rgb_frame_trans = TransformStamped()
        # world_frame2rgb_frame_trans.header.frame_id = self.world_frame_id
        # world_frame2rgb_frame_trans.header.stamp = rospy.Time.now()
        # world_frame2rgb_frame_trans.child_frame_id = self.rgb_frame_id
        # world_frame2rgb_frame_trans.transform = world_frame2rgb_frame
        # static_tf_boardcaster.sendTransform(
        #     [world_frame2rgb_frame_trans, world_frame2depth_frame_trans]
        # )

        # world_frame2depth_frame_transformation = np.eye(4)
        # world_frame2depth_frame_transformation[:3, :3] = (
        #     transforms3d.quaternions.quat2mat(
        #         [
        #             world_frame2depth_frame.rotation.w,
        #             world_frame2depth_frame.rotation.x,
        #             world_frame2depth_frame.rotation.y,
        #             world_frame2depth_frame.rotation.z,
        #         ]
        #     )
        # )
        # world_frame2depth_frame_transformation[:3, 3] = [
        #     world_frame2depth_frame.translation.x,
        #     world_frame2depth_frame.translation.y,
        #     world_frame2depth_frame.translation.z,
        # ]
        # depth_frame2world_frame = np.linalg.inv(world_frame2depth_frame_transformation)
        # world_frame2rgb_frame_transformation = np.eye(4)
        # world_frame2rgb_frame_transformation[:3, :3] = (
        #     transforms3d.quaternions.quat2mat(
        #         [
        #             world_frame2rgb_frame.rotation.w,
        #             world_frame2rgb_frame.rotation.x,
        #             world_frame2rgb_frame.rotation.y,
        #             world_frame2rgb_frame.rotation.z,
        #         ]
        #     )
        # )
        # world_frame2rgb_frame_transformation[:3, 3] = [
        #     world_frame2rgb_frame.translation.x,
        #     world_frame2rgb_frame.translation.y,
        #     world_frame2rgb_frame.translation.z,
        # ]
        # self.depth_frame2rgb_frame = np.matmul(
        #     world_frame2rgb_frame_transformation, depth_frame2world_frame
        # )
        # self.rgb_frame2depth_frame = np.linalg.inv(self.depth_frame2rgb_frame)
        # print(self.depth_frame2rgb_frame)
        self.points_frame2world_frame = None

        self.camera_info_sub_depth.unregister()
        self.camera_info_sub_rgb.unregister()
        # self.tf_world2depth_sub.unregister()

        self.intrinsic_matrix = None
        # self.manually_transfromed_pointcloud_pub = rospy.Publisher(
        #     "pointcloud_topic", PointCloud2, queue_size=10
        # )
        self.target_object_pointcloud_pub = rospy.Publisher(
            "target_object_pointcloud_topic", PointCloud2, queue_size=10
        )
        self.depth_img_sub = rospy.Subscriber(
            "/World/EC/depth",
            Image,
            self.depth_img_callback,
        )
        self.rgb_img_sub = rospy.Subscriber(
            "/World/EC/rgb",
            Image,
            self.rgb_image_callback,
        )
        self.points_sub = rospy.Subscriber(
            "/World/EC/points",
            PointCloud2,
            self.points_callback,
        )
        self.timer = rospy.Timer(rospy.Duration(1), self.predict)

        self.vlm = vlmpipeline
        self.has_processed_first_frame = False
        self.frames = None
        self.point = None
        self.category_multiplier = 100  # which means the instance number of each object categery is less than 100
        self.target_objects = ["a man who raises his hand sideways"]
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
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = ConfigurableTransformListener(
            tf_topic="/World/tf", buffer=self.tf_buffer
        )
        while not self.tf_buffer.can_transform(
            self.world_frame_id, self.rgb_frame_id, rospy.Time.now()
        ):
            rospy.sleep(1)
            rospy.loginfo(f"{rospy.Time.now()}: wait for tf message")
        self.points_frame2world_frame = self.tf_buffer.lookup_transform(
            self.world_frame_id, self.rgb_frame_id, rospy.Time.now()
        )

    def predict(self, event):
        while self.frames is None or self.point is None:
            rospy.sleep(1)
        with torch.no_grad():
            if not self.has_processed_first_frame:
                start_time_process_first_frame = time.time()
                masks = self.vlm.process_first_frame(
                    self.target_objects, self.frames, owlv2_threshold=0.12
                )
                end_time_process_first_frame = time.time()
                rospy.loginfo(
                    f"process first frame cost {end_time_process_first_frame-start_time_process_first_frame}s"
                )
                if not np.any(masks):
                    raise ValueError(
                        "no intrested object found in the scene, may be you should let robot turn around or change the scene or change the target object"
                    )
                self.has_processed_first_frame = True
                self.masks = masks
                result = self.get_3d_obs_by_name_by_vlm(self.target_objects[0])
            else:
                print("process frame")
                start_time_process_frame = time.time()
                masks = self.vlm.process_frame(self.frames)
                end_time_process_frame = time.time()
                rospy.loginfo(
                    f"process frame cost {end_time_process_frame-start_time_process_frame}s"
                )
                if not np.any(masks):
                    raise ValueError(
                        "no intrested object found in the scene, may be you should let robot turn around or change the scene or change the target object"
                    )
                self.masks = masks
                result = self.get_3d_obs_by_name_by_vlm(self.target_objects[0])
            # write masks for debug
            cv2.imwrite("masks.png", masks[0].astype(np.uint8) * 255)
            # save masked rgb image for debug
            # # Combine RGB and masks
            # combined_img = cv2.addWeighted(self.frames[0], 0.7, self.masks[0], 0.3, 0)
            # combined_img_depth = cv2.addWeighted(self.frames[0], 0.7, self.masks[0], 0.3, 0)
            # # Save the combined image
            # cv2.imwrite('combined_img.png', combined_img)
            # cv2.imwrite('combined_img_depth',combined_img_depth)

    def camera_info_callback(self, camera_info):
        K = camera_info.K
        intrinsic_matrix = np.array(K).reshape(3, 3)
        self.image_size = (camera_info.width, camera_info.height)
        self.intrinsic_matrix = intrinsic_matrix

    def rgb_image_callback(self, rgb_img):
        rgb_img = self._cvb.imgmsg_to_cv2(rgb_img)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_frame = rgb_img.transpose([2, 0, 1])
        self.frames = np.stack([rgb_frame], axis=0)

    def depth_img_callback(self, depth_img):
        depth_img = self._cvb.imgmsg_to_cv2(depth_img)
        # cv2.imwrite('/shared/codes/voxposer-ros/depth_img.png', depth_img/65535*255)
        self.depth = depth_img / 65535 * 255
        depth_img_float = depth_img.astype(np.float32) / 65535.0 * 20
        # M = (
        #     self.intrinsic_matrix_rgb
        #     @ self.depth_frame2rgb_frame[:3, :3]
        #     @ np.linalg.inv(self.intrinsic_matrix_depth)
        # )
        # depth_img_float_transformed = cv2.warpPerspective(
        #     depth_img_float, M, (self.image_size[1], self.image_size[0])
        # )
        depth_img_float_transformed = (
            depth_img_float  # no need to transform depth image
        )
        # save depth frame and transformed depth for debug, attention the depth_img_float_* should be normalized to 0-255
        depth_img_color = cv2.applyColorMap(
            (depth_img_float / np.max(depth_img_float) * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        depth_img_transformed_color = cv2.applyColorMap(
            (
                depth_img_float_transformed / np.max(depth_img_float_transformed) * 255
            ).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        cv2.imwrite("depth_img_color.png", depth_img_color)
        cv2.imwrite("depth_img_transformed.png", depth_img_transformed_color)
        cv2.imwrite("rgb_img.png", self.frames[0].transpose([1, 2, 0]))
        # self._trans_depth_to_pcd(depth_img_float_transformed, None, None, None, None)

    def points_callback(self, data: PointCloud2):
        if self.points_frame2world_frame is None:
            return
        point_world_frame: PointCloud2 = do_transform_cloud(
            data, self.points_frame2world_frame
        )
        self.point = np.frombuffer(point_world_frame.data, dtype=np.float32).reshape(
            -1, 3
        )

    def _trans_depth_to_pcd(
        self, depth_img, image_size, FOV, camera_rotation, camera_position
    ):
        intrinsic_matrix = self.intrinsic_matrix_rgb
        image_size = self.image_size
        uv_matrix = np.mgrid[0 : image_size[0], 0 : image_size[1]].reshape(2, -1)
        z_vector = depth_img.reshape(-1)  # fuck opencv, i dont kown why
        # z_vector[z_vector > 3] = 0
        zuv_matrix = uv_matrix * z_vector.reshape(-1)
        zuvz_matrix = np.concatenate([zuv_matrix, z_vector.reshape(1, -1)], axis=0)
        point_camera_frame = np.matmul(
            np.linalg.inv(intrinsic_matrix), zuvz_matrix
        )  # use inverse of intrinsic matrix to get the point in camera frame

        pointcloud = PointCloud2()
        pointcloud.header = Header(stamp=rospy.Time.now(), frame_id=self.rgb_frame_id)
        pointcloud.height = 1
        pointcloud.width = point_camera_frame.shape[1]
        pointcloud.fields = [
            PointField(name="x", offset=0, datatype=7, count=1),
            PointField(name="y", offset=4, datatype=7, count=1),
            PointField(name="z", offset=8, datatype=7, count=1),
        ]
        pointcloud.is_bigendian = False
        pointcloud.point_step = 12
        pointcloud.row_step = 12 * pointcloud.width
        pointcloud.is_dense = True
        pointcloud.data = np.array(point_camera_frame.T, dtype=np.float32).tostring()
        self.point = point_camera_frame.T
        self.manually_transfromed_pointcloud_pub.publish(pointcloud)

    def publish_target_object_pointcloud(self, pointcloud_):
        pointcloud = PointCloud2()
        pointcloud.header = Header(stamp=rospy.Time.now(), frame_id=self.rgb_frame_id)
        pointcloud.height = 1
        pointcloud.width = pointcloud_.shape[0]
        pointcloud.fields = [
            PointField(name="x", offset=0, datatype=7, count=1),
            PointField(name="y", offset=4, datatype=7, count=1),
            PointField(name="z", offset=8, datatype=7, count=1),
        ]
        pointcloud.is_bigendian = False
        pointcloud.point_step = 12
        pointcloud.row_step = 12 * pointcloud.width
        pointcloud.is_dense = True
        pointcloud.data = np.array(pointcloud_, dtype=np.float32).tostring()
        self.target_object_pointcloud_pub.publish(pointcloud)

    def get_3d_obs_by_name_by_vlm(self, query_name, cameras=None):
        """
        Retrieves 3D point cloud observations and normals of an object by its name by VLM

        Args:
            query_name (str): The name of the object to query.
            cameras (list): list of camera names, if None, use all cameras
        Returns:
            tuple: A tuple containing object points and object normals.
        """
        # assert query_name in self.target_objects, f"Unknown object name: {query_name}"
        points, masks = [], []
        mask_frame = self.masks[0]
        point = self.point
        points.append(point.reshape(-1, 3))
        masks.append(
            mask_frame.reshape(-1)
        )  # it contain the mask of different type of object
        points = np.concatenate(points, axis=0)
        masks = np.concatenate(masks, axis=0)  # [0,101,102,201,202,0,0,0,301]

        categery_masks = (
            masks.astype(np.int32) // self.category_multiplier
        )  # [0,1,1,2,2,0,0,0,3]
        # get object points
        category_label = self.name2categerylabel[query_name]  # 1
        # objs_mask: [0,101,102,0,0,0,0,0,0]
        masks[~np.isin(categery_masks, category_label)] = 0  # [0,101,102,0,0,0,0,0,0]
        if not np.any(masks):
            # which masks == [0,0,0,0,0,0,0,0,0] if category_label == 4
            print(f"Object {query_name} not found in the scene")
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
            obj_points = points[obj_mask]
            # obj_normals = normals[obj_mask]
            # voxel downsample using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj_points)
            # pcd.normals = o3d.utility.Vector3dVector(obj_normals)
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.001)
            pcd_downsampled_filted, ind = pcd_downsampled.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=1.0
            )
            obj_points = np.asarray(pcd_downsampled_filted.points)
            # obj_normals = np.asarray(pcd_downsampled_filted.normals)
            objs_points.append(obj_points)
            # save pcd to file for debug
            o3d.io.write_point_cloud(
                f"{query_name}_{obj_ins_id}.pcd", pcd_downsampled_filted
            )
            masked_rgb = self.frames[0].transpose([1, 2, 0]).copy()
            masked_rgb[masks[0] == 0] = 0
            cv2.imwrite("masked_rgb.png", masked_rgb)
            obj_masked_rgb = self.frames[0].transpose([1, 2, 0]).copy()
            obj_masked_rgb[~obj_mask.reshape(self.image_size)] = 0
            cv2.imwrite(f"{query_name}_{obj_ins_id}_masked_rgb.png", obj_masked_rgb)
            cv2.imwrite(
                f"{query_name}_{obj_ins_id}_mask.png",
                obj_mask.reshape(self.image_size).astype(np.uint8) * 255,
            )
            self.publish_target_object_pointcloud(obj_points)
        print(f"we find {len(objs_points)} instances of {query_name}")
        # calculate the 3d object height width and distance
        x_min = np.min(objs_points[0][:, 0])
        x_max = np.max(objs_points[0][:, 0])
        y_min = np.min(objs_points[0][:, 1])
        y_max = np.max(objs_points[0][:, 1])
        z_min = np.min(objs_points[0][:, 2])
        z_max = np.max(objs_points[0][:, 2])
        width = x_max - x_min
        height = y_max - y_min
        distance = (z_max + z_min) / 2
        print(
            f"{query_name}: width: {width}m, height: {height}m, distance: {distance}m"
        )

        return objs_points

    def run(self):
        # while self.intrinsic_matrix is None:
        #     rospy.sleep(1)
        # Call the _trans_depth_to_pcd method with appropriate arguments
        # depth_img = ...
        # image_size = ...
        # FOV = ...
        # camera_rotation = ...
        # camera_position = ...
        # pointcloud = self._trans_depth_to_pcd(depth_img, image_size, FOV, camera_rotation, camera_position)

        # # Publish the pointcloud
        # self.publisher.publish(pointcloud)

        # Spin the node
        rospy.spin()


if __name__ == "__main__":
    with torch.no_grad():
        owlv2_model_path = "/models/google-owlv2-large-patch14-finetuned"
        owlv2_model_path = "/models/google-owlv2-base-patch16-ensemble"
        sam_model_path = "/models/facebook-sam-vit-huge"
        # sam_model_path = "/models/facebook-sam-vit-base"
        xmem_model_path = "/models/XMem.pth"
        resnet_18_path = "/models/resnet18.pth"
        resnet_50_path = "/models/resnet50.pth"
        vlmpipeline = VLM(
            owlv2_model_path,
            sam_model_path,
            xmem_model_path,
            resnet_18_path,
            resnet_50_path,
            verbose=False,
            resize_to=[480, 480],
            verbose_frame_every=1,
            input_batch_size=1,
        )

        node = DepthToPointCloudNode(vlmpipeline=vlmpipeline)
        node.run()
