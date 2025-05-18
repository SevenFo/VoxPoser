#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理类，处理应用程序配置
"""

import yaml
import os
import logging
from copy import deepcopy


class ConfigManager:
    """配置管理类，负责加载和保存应用程序配置"""

    DEFAULT_CONFIG = {
        "app": {
            "headless": False,
            "log_level": "DEBUG",
            "window_width": 1280,
            "window_height": 720,
        },
        "robot": {
            "usd_path": "C:/Users/hp/Documents/Projects/UR5IsaacSim/usd/ur5_with_robotiq_gripper/ur5_with_gripper_single_articulation_root.usd",
            "use_gripper": True,
            "joint_names_without_gripper": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            "gripper_joint_names": ["finger_joint", "right_outer_knuckle_joint"],
            "end_effector_prim_name": "base_link_gripper",
            # "initial_joints": [112, -100, 125, -120, -90, 0],
            "initial_joints": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "base_position": [0.85, -7.0, 1.18],
        },
        "task": {
            "name": "FollowTarget",
            "target_position": [0.5, 0, 0.4],
            "scene_path": "C:/Users/hp/Documents/Projects/UR5IsaacSim/scenes/Collected_ROOM_set_fix_0401/ROOM_set.usd",
        },
        "mocap": {"udp_port": 8002, "model_position": [0.3, 1.0, 0]},
        "rtde": {"ip": "localhost", "port": 30004, "enabled": False},
        "ui": {
            "frame_time": 1.0,
            "follow_time": 0.1,
            "gain": 300,
            "gripping_threshold": 0.05,
        },
    }

    def __init__(self, config_path=None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径，如果为None，则使用默认路径
        """
        # 设置默认配置文件路径
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.config_path = os.path.join(base_dir, "conf", "app_config.yaml")
        else:
            self.config_path = config_path

        # 设置配置内容
        self._configs = {}

        # 设置日志器
        self.logger = logging.getLogger(__name__)

        # 加载配置
        self.load_config()

    def load_config(self):
        """加载配置文件，如果文件不存在则创建默认配置"""
        try:
            if os.path.exists(self.config_path):
                self.logger.info(f"加载配置文件: {self.config_path}")
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._configs = yaml.safe_load(f) or {}
                return True
            else:
                self.logger.warning(f"配置文件不存在，将创建默认配置: {self.config_path}")
                return self._create_default_config()
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return False

    def save_config(self):
        """保存配置到文件"""
        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._configs, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"配置已保存到: {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
            return False

    def _create_default_config(self):
        """创建默认配置"""
        try:
            # 使用默认配置
            self._configs = deepcopy(self.DEFAULT_CONFIG)

            # 保存默认配置
            return self.save_config()
        except Exception as e:
            self.logger.error(f"创建默认配置失败: {e}")
            return False

    def set_value(self, section, option, value):
        """
        设置配置值

        Args:
            section: 配置节
            option: 配置项
            value: 配置值

        Returns:
            是否成功
        """
        try:
            if section not in self._configs:
                self._configs[section] = {}

            self._configs[section][option] = value
            return True
        except Exception as e:
            self.logger.error(f"设置配置值失败 [{section}.{option}]: {e}")
            return False

    def get_section_config(self, section):
        """
        获取指定配置节的所有配置，使用默认值作为备选

        Args:
            section: 配置节名称

        Returns:
            配置字典
        """
        if section not in self.DEFAULT_CONFIG:
            self.logger.warning(f"请求的配置节不存在: {section}")
            return {}

        section_config = self._configs.get(section, {})
        default_section = self.DEFAULT_CONFIG[section]

        # 基于默认配置合并用户配置
        result = {}
        for key in default_section:
            result[key] = section_config.get(key, default_section[key])

        return result

    # 以下是特定配置获取方法，全部基于通用方法实现
    def get_app_config(self):
        """获取应用程序配置"""
        return self.get_section_config("app")

    def get_robot_config(self):
        """获取机器人配置"""
        return self.get_section_config("robot")

    def get_mocap_config(self):
        """获取动作捕捉配置"""
        return self.get_section_config("mocap")

    def get_rtde_config(self):
        """获取RTDE配置"""
        return self.get_section_config("rtde")

    def get_ui_config(self):
        """获取UI配置"""
        return self.get_section_config("ui")

    @property
    def configs(self):
        """获取配置"""
        return self._configs
