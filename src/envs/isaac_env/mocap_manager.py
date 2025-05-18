#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作捕捉管理类，负责处理动作捕捉相关功能
"""

import numpy as np
import logging
from noitom.axis.mocap.client import MocapClient as BaseMocapClient
from noitom.axis.mocap.client_no_scene import MocapClient as BaseMocapClientNoScene


class MocapManager(BaseMocapClientNoScene):
    def __init__(self, config_manager=None, udp_port=8003, prim_prefix=""):
        """
        初始化动作捕捉管理器

        Args:
            config_manager: 配置管理器实例
            udp_port: UDP端口号，如果提供了配置管理器，则从配置中读取
        """
        # 设置日志器
        self.logger = logging.getLogger(__name__)

        # 从配置中读取参数
        self.config_manager = config_manager
        if config_manager:
            mocap_config = config_manager.get_mocap_config()
            udp_port = mocap_config.get("udp_port", udp_port)

        # 初始化基类
        try:
            super().__init__(udp_port, prim_prefix)
            self.logger.info(f"动作捕捉管理器初始化完成，UDP端口：{udp_port}")
        except Exception as e:
            self.logger.error(f"初始化动作捕捉管理器失败: {e}")

        self.actor_data = None

    def setup_scene(self, world=None):
        """
        设置场景

        Returns:
            成功返回True，失败返回False
        """
        # try:
        super().setup_scene()
        self.logger.info("动作捕捉场景设置完成")
        return True
        # return True
        # except Exception as e:
        #     self.logger.error(f"设置动作捕捉场景失败: {e}")
        #     return False

    def move_model_to_origin(self):
        """
        移动模型到原点

        Returns:
            成功返回True，失败返回False
        """
        try:
            # 从配置中获取模型位置
            if self.config_manager:
                mocap_config = self.config_manager.get_mocap_config()
                model_position = mocap_config.get("model_position")
                if model_position and len(model_position) == 7:
                    super().move_model(*model_position)
                    return True
            # 如果没有配置，则默认移动到原点
            else:
                super().move_model(0, 0, 0, 1, 0, 0, 0)
            self.logger.info("已将动作捕捉模型移动到原点")
            return True
        except Exception as e:
            self.logger.error(f"移动动作捕捉模型到原点失败: {e}")
            return False

    def move_model(self, x=None, y=None, z=None, qw=None, qx=None, qy=None, qz=None):
        """
        移动模型位置

        Args:
            x, y, z: 模型位置坐标

        Returns:
            成功返回True，失败返回False
        """
        try:
            super().move_model(x, y, z, qw, qx, qy, qz)
            self.logger.info(
                f"已移动动作捕捉模型到位置: ({x}, {y}, {z}, {qw}, {qx}, {qy}, {qz})"
            )
            return True
        except Exception as e:
            self.logger.error(f"移动动作捕捉模型失败: {e}")
            return False

    def update_avatar_posture(self, actor_id=1):
        """
        更新虚拟角色姿态

        Args:
            actor_id: 角色ID

        Returns:
            成功返回True，失败返回False
        """
        try:
            self.updateAvatarPosture(actor_id)
            return True
        except Exception as e:
            self.logger.error(f"更新角色姿态失败: {e.__class__.__name__}: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_avatar_joints_data(self):
        """
        获取角色关节数据

        Returns:
            关节数据字典，失败返回空字典
        """
        try:
            return super().get_avatar_joints_data()
        except Exception as e:
            self.logger.error(f"获取角色关节数据失败: {e}")
            return {}

    def get_world_avatar_data(self):
        """
        获取世界坐标系中的角色数据

        Returns:
            角色数据字典，失败返回空字典
        """
        try:
            return super().get_world_avatar_data()
        except Exception as e:
            self.logger.error(f"获取世界坐标系中的角色数据失败: {e}")
            return {}

    def get_world_avatar_quaternion(self):
        """
        获取世界坐标系中的角色四元数数据, quat format: [x, y, z, w]

        Returns:
            角色四元数数据字典，失败返回空字典
        """
        try:
            avatar_data = self.get_world_avatar_data()
            return {
                bone_name: data.get("quat", [1, 0, 0, 0])
                for bone_name, data in avatar_data.items()
            }
        except Exception as e:
            self.logger.error(f"获取世界坐标系中的角色四元数数据失败: {e}")
            return {}

    def get_bone_names(self):
        """
        获取骨骼名称列表

        Returns:
            骨骼名称列表，失败返回空列表
        """
        try:
            return self.get_bone_name_dic()
        except Exception as e:
            self.logger.error(f"获取骨骼名称列表失败: {e}")
            return []

    def has_bone_data(self):
        """
        检查是否有骨骼数据

        Returns:
            有骨骼数据返回True，否则返回False
        """
        try:
            return super().has_bone_data()
        except Exception as e:
            self.logger.error(f"检查骨骼数据失败: {e}")
            return False

    def get_world_bone_data(self, bone_name):
        """
        获取世界坐标系中的骨骼数据

        Args:
            bone_name: 骨骼名称

        Returns:
            骨骼数据字典，失败返回空字典
        """
        try:
            return super().get_world_bone_data(bone_name)
        except Exception as e:
            self.logger.error(f"获取世界坐标系中的骨骼数据失败: {e}")
            return {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]}

    def euler_to_quaternion(self, euler_angles, order="yxz"):
        """
        欧拉角转四元数

        Args:
            euler_angles: 欧拉角数组
            order: 旋转顺序

        Returns:
            四元数数组
        """
        try:
            return self.to_quat(euler_angles, order)
        except Exception as e:
            self.logger.error(f"欧拉角转四元数失败: {e}")
            return [1, 0, 0, 0]  # 返回单位四元数

    def quaternion_to_euler(self, quaternion, degrees=True, order="yxz"):
        """
        四元数转欧拉角

        Args:
            quaternion: 四元数数组
            degrees: 是否返回角度制
            order: 旋转顺序

        Returns:
            欧拉角数组
        """
        try:
            return self.to_euler(quaternion, degrees, order)
        except Exception as e:
            self.logger.error(f"四元数转欧拉角失败: {e}, input: {quaternion}")
            import traceback

            traceback.print_exc()
            return [0, 0, 0]

    def eulers_inverse(self, pre_angles, current_angles):
        """
        解决欧拉角跳变问题(欧拉角数组)

        Args:
            pre_angles: 前一时刻欧拉角数组
            current_angles: 当前欧拉角数组

        Returns:
            处理后的欧拉角数组
        """
        try:
            return super().eulers_inverse(pre_angles, current_angles)
        except Exception as e:
            self.logger.error(f"解决欧拉角数组跳变问题失败: {e}")
            return current_angles if current_angles is not None else [0, 0, 0]

    def euler_inverse(self, pre_angle, current_angle):
        """
        解决欧拉角跳变问题(单个欧拉角)

        Args:
            pre_angle: 前一时刻欧拉角
            current_angle: 当前欧拉角

        Returns:
            处理后的欧拉角
        """
        try:
            return super().euler_inverse(pre_angle, current_angle)
        except Exception as e:
            self.logger.error(f"解决单个欧拉角跳变问题失败: {e}")
            return current_angle if current_angle is not None else 0

    def quaternion_multiply(self, q1, q2):
        """
        四元数乘法

        Args:
            q1, q2: 要相乘的两个四元数

        Returns:
            相乘后的四元数
        """
        try:
            return self.q_multi(q1, q2)
        except Exception as e:
            self.logger.error(f"四元数乘法失败: {e}")
            return [1, 0, 0, 0]  # 返回单位四元数

    def quaternion_from_axis_angle(self, axis, angle_rad):
        """
        从轴角表示转换为四元数

        Args:
            axis: 旋转轴
            angle_rad: 旋转角度(弧度)

        Returns:
            四元数
        """
        try:
            axis = np.asarray(axis)
            axis = axis / np.linalg.norm(axis)  # 归一化
            half_angle = angle_rad / 2.0
            sin_half_angle = np.sin(half_angle)
            return np.array(
                [
                    np.cos(half_angle),
                    axis[0] * sin_half_angle,
                    axis[1] * sin_half_angle,
                    axis[2] * sin_half_angle,
                ]
            )
        except Exception as e:
            self.logger.error(f"从轴角表示转换为四元数失败: {e}")
            return [1, 0, 0, 0]  # 返回单位四元数

    def get_relative_trans(self):
        return super().get_relative_trans()
