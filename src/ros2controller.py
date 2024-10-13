import numpy as np
from envs.ros_env.ros2_drone_env import VoxPoserROS2DroneEnv
from pyproj import Transformer
import logging
from pymavlink import mavutil
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleMavlinkController:
    def __init__(self, env, config) -> None:
        assert type(env) == VoxPoserROS2DroneEnv or True, "SimpleMavlinkController only works with VoxPoserROS2DroneEnv"
        
        self.env = env
        self.config = config
        
        self.connection_string = self.config.get("connection_string", "udp:localhost:14540")
        self.master = None
        self.is_armed = False
        self.mission_item_reached = None
            
    def connect(self):
        if self.master is None:
            try:
                self.master = mavutil.mavlink_connection(self.connection_string)
                self.master.wait_heartbeat()
                logging.info("Heartbeat from system (system %u component %u)" % (self.master.target_system, self.master.target_component))
                self.get_initial_parameters()
            except Exception as e:
                logging.error(f"Failed to connect to MAVLink: {e}")
        else:
            logging.info("Already connected to MAVLink")

    def get_initial_parameters(self):
        """
        获取当前的详细参数，包括坐标、初始位置等
        """
        try:
            # 获取当前坐标
            self.lat0, self.lon0, self.alt0, self.heading0 = self.get_current_position()
            logging.info(f"Current Position - Latitude: {self.lat0}, Longitude: {self.lon0}, Altitude: {self.alt0}, Heading: {self.heading0}")
            
            # 获取其他详细参数
            base_mode, custom_mode = self.get_current_mode()
            logging.info(f"Current Mode - Base Mode: {base_mode}, Custom Mode: {custom_mode}")
            
        except Exception as e:
            logging.error(f"Failed to get initial parameters: {e}")


    def get_current_position(self):
        """
        获取飞行器的当前位置（经度、纬度、高度）和航向角
        """
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg:
            latitude = msg.lat / 1e7
            longitude = msg.lon / 1e7
            altitude = msg.alt / 1e3  # 相对高度，单位米
            heading = msg.hdg / 100.0  # 航向角，单位为度
            return latitude, longitude, altitude, heading
        return None, None, None, None

    def arm(self):
        if self.is_armed:
            logging.info("already armed")
            return True
        try:
            self.master.arducopter_arm()
            logging.info("Arming motors...")

            # Wait for ACK
            ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=10)
            if ack is None:
                logging.error("No ACK received for arm command")
                return False
            logging.debug(f"Receive ACK command: {ack.command}")
            if ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                self.master.motors_armed_wait()
                logging.info("Motors armed")
                self.is_armed = True
                return True
            else:
                logging.error(f"Arm command failed with result: {ack.result}")
                return False

        except Exception as e:
            logging.error(f"Failed to arm motors: {e}")
            return False
        
    def takeoff(self, altitude=3.0):
        try:
            if not self.is_armed:
                logging.warning("not arming")
                return False
            
            self.master.mav.command_int_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,0,
                0, 0, 0, 0, int(self.lat0*1e7), int(self.lon0*1e7), int(altitude*1e3)
            )
            logging.info(f"Taking off to altitude {[self.lat0, self.lon0, self.alt0 + altitude]}")

            # Wait for ACK
            ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=10)
            if ack is None:
                logging.error("No ACK received for takeoff command")
                # return False

            if ack.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                logging.debug("Takeoff command accepted")
                # return True
            else:
                logging.error(f"Takeoff command failed with result: {ack.command}: {ack.result}")
                # return False
            
            while True:
                # _,_, alt,heading = self.get_current_position()
                
                msg = self.master.recv_match(type=['GLOBAL_POSITION_INT','COMMAND_ACK','STATUSTEXT','COMMAND_ACK'], blocking=True)
                if msg:
                    logging.debug(f"RECEIVE MSG: {msg.to_dict()}")
                    if msg.get_type() == 'GLOBAL_POSITION_INT':
                        if msg.relative_alt >= 3.0 * 1e3:
                            logging.info("Reached target altitude")
                            break
            return True

        except Exception as e:
            logging.error(f"Failed to takeoff: {e}")
            return False

    def change_mode(self, mode, custom_mode=None, custom_sub_mode=None, timeout=10):
        """
        改变 PX4 飞行模式并等待 ACK
        :param mode: 目标飞行模式
        :param custom_mode: 自定义模式（可选）
        :param custom_sub_mode: 自定义子模式（可选）
        :param timeout: 等待 ACK 的超时时间（秒）
        :return: 如果模式改变成功返回 True，否则返回 False
        """
        self.master.set_mode_px4(mode=mode, custom_mode=custom_mode, custom_sub_mode=custom_sub_mode)
        logging.info(f"Setting mode to {mode}")

        # 等待 ACK
        ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=timeout)
        if ack is None:
            logging.error("No ACK received for mode change command")
            return False

        if ack.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            logging.debug("Mode change command accepted")
            return True
        else:
            logging.error(f"Mode change command failed with result: {ack.command}: {ack.result}")
            return False
        
    def autotakeoff(self):
        try:
            if not self.is_armed:
                logging.warning("not arming")
                return False

            # 设置飞行模式为 AUTO.TAKEOFF
            if not self.change_mode(mode="TAKEOFF"):
                logging.error("Failed to change mode to TAKEOFF")
                return False

            # 等待飞行器达到目标高度的状态消息
            while True:
                current_mode = self.get_current_mode()
                if mavutil.interpret_px4_mode(*current_mode) == "LOITER":
                    logging.info("Completed takeoff, return to LOITER mode")
                    return True

        except Exception as e:
            logging.error(f"Failed to takeoff: {e}")
            return False

    def land(self):
        try:
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND, 0,
                0, 0, 0, 0, 0, 0, 0
            )
            logging.info("Landing command sent")

            # 检查ACK消息，确认命令是否被接受
            ack_received = False
            while True:
                ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
                if ack_msg and ack_msg.command == mavutil.mavlink.MAV_CMD_NAV_LAND:
                    if ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                        logging.debug("Landing command accepted")
                        ack_received = True
                        break
                    else:
                        logging.error(f"Landing command rejected with result: {ack_msg.result}")
                        return

            if not ack_received:
                logging.error("No ACK received for landing command")
                return

            # 等待飞行器完全降落
            while True:
                msg = self.master.recv_match(type=['EXTENDED_SYS_STATE'],blocking=False)
                if not msg:
                    continue

                if msg.get_type() == 'EXTENDED_SYS_STATE':
                    if msg.landed_state == mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND:
                        logging.debug("Landed state: ON_GROUND")
                        logging.info("Landing complete")
                        break
                    continue
                
                logging.info(f"RECEIVE MSG: {msg.get_type()}")
        except Exception as e:
            logging.error(f"Failed to land: {e}")
            
    def disarm(self):
        try:
            self.master.arducopter_disarm()
            self.master.motors_disarmed_wait()
            logging.info("Motors disarmed")
        except Exception as e:
            logging.error(f"Failed to disarm motors: {e}")
            
    def get_current_mode(self):
        """
        获取当前的飞行模式
        """
        try:
            msg = self.master.recv_match(type='HEARTBEAT', blocking=True)
            if msg:
                base_mode = msg.base_mode
                custom_mode = msg.custom_mode
                mode_string = mavutil.interpret_px4_mode(base_mode, custom_mode)
                logging.debug(f"Current base mode: {base_mode}, custom mode: {custom_mode}")
                logging.debug(f"Current Base Mode: {self._parse_base_mode(base_mode)}")
                logging.info(f"Current mode_string: {mode_string}")
                return base_mode, custom_mode
            else:
                logging.error("Failed to receive HEARTBEAT message")
                return None, None
        except Exception as e:
            logging.error(f"Failed to get current mode: {e}")
            return None, None
        
    def _parse_base_mode(self, base_mode):
        modes = {
            0: "CUSTOM_MODE_ENABLED",
            1: "TEST_ENABLED",
            2: "AUTO_ENABLED",
            3: "GUIDED_ENABLED",
            4: "STABILIZE_ENABLED",
            5: "HIL_ENABLED",
            6: "MANUAL_INPUT_ENABLED",
            7: "SAFETY_ARMED"
        }
        
        binary_representation = bin(base_mode)[2:].zfill(8)
        active_modes = [modes[i] for i in range(8) if binary_representation[7 - i] == '1']
        
        return active_modes
        
    def clear_mission(self):
        try:
            self.master.mav.mission_clear_all_send(self.master.target_system, self.master.target_component)
            logging.info("清除所有任务")

            # 等待ACK
            ack_received = False
            while not ack_received:
                ack_msg = self.master.recv_match(type = ['MISSION_ACK'],blocking=True, timeout=5)
                if ack_msg:
                    if ack_msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                        logging.info("任务清除成功")
                        ack_received = True
                    else:
                        logging.error(f"任务清除失败：MISSION_ACK: {ack_msg.to_dict()}")
                else:
                    logging.error("任务清除失败，未收到ACK")
                    break
        except Exception as e:
            logging.error(f"清除任务失败: {e}")
        
    def add_waypoint(self, seq, frame, command, current, autocontinue, param1, param2, param3, param4, latitude, longitude, altitude):
        try: 
            self.master.mav.mission_item_send(
                self.master.target_system,
                self.master.target_component,
                seq,
                frame,
                command,
                current,
                autocontinue,
                param1,
                param2,
                param3,
                param4,
                latitude,
                longitude,
                altitude
            )
            logging.info(f"添加路径点: seq={seq}, latitude={latitude}, longitude={longitude}, altitude={altitude}")
        except Exception as e:
            logging.error(f"添加路径点失败: {e}")
            
    def _calculate_yaw(self, lat1, lon1, lat2, lon2):
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        yaw = math.degrees(math.atan2(delta_lon, delta_lat))
        if yaw < 0:
            yaw += 360
        return yaw

    def set_mission(self, waypoints):
        self.clear_mission()
        lat, lon, alt, heading = self.get_current_position()
        try:
            logging.debug("Try to set mission count = {}".format(len(waypoints)))
            self.master.mav.mission_count_send(self.master.target_system, self.master.target_component, len(waypoints))
            # 等待MISSION_REQUEST的ACK
            while True:
                ack_msg = self.master.recv_match(type='MISSION_REQUEST', blocking=True, timeout=5)
                if ack_msg:
                    if ack_msg.mission_type == mavutil.mavlink.MAV_MISSION_TYPE_MISSION:
                        logging.debug(f"MISSION_REQUEST received for seq: {ack_msg.seq}")
                        seq = ack_msg.seq
                        if seq < len(waypoints):
                            waypoint = waypoints[seq]
                            
                            if seq == 0:
                                # 第一个目标点基于当前位置和目标位置计算
                                yaw = self._calculate_yaw(lat, lon, waypoint[0], waypoint[1])
                            else:
                                # 后续目标点基于当前目标点和下一个目标点计算
                                prev_waypoint = waypoints[seq - 1]
                                yaw = self._calculate_yaw(prev_waypoint[0], prev_waypoint[1], waypoint[0], waypoint[1])

                            self.add_waypoint(
                                seq=seq,
                                frame=mavutil.mavlink.MAV_FRAME_GLOBAL,
                                command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                                current=0,
                                autocontinue=1,
                                # hold time (s), acceptance radius (m), pass radius, yaw, latitude, longitude, altitude
                                param1=0, param2=0.1, param3=0, param4=yaw, 
                                latitude=waypoint[0], longitude=waypoint[1], altitude=waypoint[2]
                            )
                        else:
                            logging.error(f"Invalid seq: {seq}")
                            return
                    else:
                        logging.error(f"Unexpected MISSION_REQUEST type: {ack_msg.mission_type}")
                        return
                else:
                    logging.warning("NO MORE MISSION_REQUEST, 可能已经全部设置完成!")
                    return
            
            logging.info("任务设置完成")
        except Exception as e:
            logging.error(f"任务设置失败: {e}")

    def start_mission(self):
        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_MISSION_START,
                0,
                0, len(self.waypoints), 0, 0, 0, 0, 0
            )
            logging.info("任务启动")

            # 等待ACK
            ack_received = False
            while not ack_received:
                ack_msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
                if ack_msg:
                    if ack_msg.command == mavutil.mavlink.MAV_CMD_MISSION_START and ack_msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                        logging.info("任务启动成功")
                        ack_received = True
                    else:
                        logging.error(f"任务启动失败，ACK类型: {ack_msg.result}")
                        break
                else:
                    logging.error("任务启动失败，未收到ACK")
                    break
        except Exception as e:
            logging.error(f"任务启动失败: {e}")
            
    def monitor_mission(self, blocking = False):
        """
        Monitor the mission progress and print the current and reached mission items.
        """
        while True and blocking:
            msg = self.master.recv_match(type=['MISSION_CURRENT', 'MISSION_ITEM_REACHED'], blocking=True, timeout=2)
            if not msg:
                continue

            if msg.get_type() == 'MISSION_CURRENT':
                self.current_mission_item = msg.seq
                logging.info(f"当前任务项: {self.current_mission_item}: {self.waypoints[self.current_mission_item]}")

            elif msg.get_type() == 'MISSION_ITEM_REACHED':
                self.mission_item_reached = msg.seq
                logging.info(f"已到达任务项: {self.mission_item_reached}")
                
            if self.mission_item_reached and self.mission_item_reached >= len(self.waypoints):
                logging.info("任务已完成")
                self.waypoints = None
                self.mission_item_reached = None
                break

    def start_mission_util_finished(self, waypoints):
        """
        Start the mission and monitor its progress until completion.
        """
        self.waypoints = waypoints
        self.set_mission(waypoints)
        self.start_mission()
        self.get_current_mode()
        self.monitor_mission(blocking=True)
    

    def auto_start_mission(self, waypoints):
        """
        自动开始任务并监控其进度，直到完成。
        """
        self.waypoints = waypoints
        self.set_mission(waypoints)
        
        # 改变飞行模式为 AUTO
        if not self.change_mode(mode="MISSION"):
            logging.error("Failed to change mode to MISSION")
            return
        
        # 获取当前模式
        current_mode = self.get_current_mode()
        current_mode = mavutil.interpret_px4_mode(*current_mode)  

        # 监控任务进度
        while current_mode == "MISSION":
            self.monitor_mission()
            current_mode = self.get_current_mode()  
            current_mode = mavutil.interpret_px4_mode(*current_mode)  
        
        logging.info(f"Mode change from MISSION to {current_mode}, MISSIONS COMPLETED")
        
    def execute(self, movable_obs, target, is_object_centric=False):
        """
        """
        assert movable_obs == None, "SimpleMavlinkController does not support movable objects"
        object_centric = is_object_centric
        if object_centric:
            raise NotImplementedError(
                "not implement execute when movable is not qudacopter"
            )
        execution_info = None
        traj = [a[0] for a in target]
        logging.info(f"Received trajectory: {traj}")
        logging.warning("Please confirm the trajectory is represented in FLU coordinate system")
        return execution_info

    def _enu_to_lla(self, xe, ye, ze, lat0, lon0, alt0):
        """
        将 ENU 坐标转换为 LLA 坐标

        参数:
        xe, ye, ze: ENU 坐标 (米)
        lat0, lon0, alt0: 参考点的地理坐标 (纬度, 经度, 高度)

        返回:
        lat, lon, alt: 转换后的地理坐标 (纬度, 经度, 高度)
        """
        # 定义 WGS-84 椭球体
        transformer_lla_to_ecef = Transformer.from_crs("epsg:4979", "epsg:4978")
        transformer_ecef_to_lla = Transformer.from_crs("epsg:4978", "epsg:4979")


        # 将参考点 LLA 转换为 ECEF 坐标
        x0, y0, z0 = transformer_lla_to_ecef.transform(xx = lat0, yy = lon0, zz = alt0)

        # 计算旋转矩阵
        slat = np.sin(np.radians(lat0))
        clat = np.cos(np.radians(lat0))
        slon = np.sin(np.radians(lon0))
        clon = np.cos(np.radians(lon0))

        R = np.array([[-slon, clon, 0],
                    [-clon*slat, -slon*slat, clat],
                    [clon*clat, slon*clat, slat]])

        # 将 ENU 坐标转换为 ECEF 坐标
        ecef_offset = np.dot(R.T, np.array([xe, ye, ze]))
        x = x0 + ecef_offset[0]
        y = y0 + ecef_offset[1]
        z = z0 + ecef_offset[2]

        # 将 ECEF 坐标转换回 LLA 坐标
        lat, lon, alt = transformer_ecef_to_lla.transform(x, y, z)

        return lat, lon, alt
    
    def _flu_to_enu(self, x, y, z, heading_rad):
        heading_rad = heading_rad - np.pi/2
        # 计算旋转矩阵
        ## 假设 heading = 90, flu = [5,0,0] -> enu = [0, -5, 0] R = [[0, -1], [1, 0]]
        R = np.array([
            [math.cos(heading_rad), +math.sin(heading_rad), 0],
            [-math.sin(heading_rad), math.cos(heading_rad), 0],
            [0, 0, 1]
        ])
        
        # 进行坐标转换
        result = R @ np.array([x, y, z])
        
        # 将结果转换为一维数组
        result = result.flatten()
        
        return result
    

    def flu_to_global_relative(self, x, y, z):
        """
        将本地坐标系（FLU）转换为全绝对高度坐标系（MAV_FRAME_GLOBAL）
        """
        current_lat, current_lon, current_alt, heading = self.get_current_position()
        if current_lat is None or current_lon is None or current_alt is None or heading is None:
            logging.error("无法获取当前飞行器位置")
            return None, None, None

        # 将航向角转换为弧度
        heading_rad = math.radians(heading)

        # 将本地坐标系（FLU）转换为 ENU 坐标系
        xe, ye, ze = self._flu_to_enu(x, y, z, heading_rad)
        
        logging.debug(f"heading: {heading}, FLU: {[x,y,z]} -> ENU :{[xe,ye,ze]}")

        # 将 ENU 坐标转换为 LLA 坐标
        new_lat, new_lon, new_alt = self._enu_to_lla(xe, ye, ze, current_lat, current_lon, current_alt)

        return new_lat, new_lon, new_alt
    
if __name__ == "__main__":
    c = SimpleMavlinkController(None, {})
    
    # 示例使用
    lat0, lon0, alt0 = 37.7749, -122.4194, 30  # 参考点的地理坐标 (例如旧金山)
    xe, ye, ze = 1000, 2000, 50  # ENU 坐标

    lat, lon, alt = c._enu_to_lla(xe, ye, ze, lat0, lon0, alt0)
    print(f'Converted LLA Coordinates: lat = {lat:.6f}, lon = {lon:.6f}, alt = {alt:.3f}')
    # Converted LLA Coordinates: lat = 37.792918, lon = -122.408047, alt = 80.393
    # passed
    xe = 5.0
    ye = 0.0
    ze = 0.0
    
    xf, yf, zf = c._flu_to_enu(xe,ye,ze,math.radians(90))
    print(xf,yf,zf)
        
    c.connect()
    c.arm()
    c.get_current_mode()
    
    # 起飞到5米高度
    if c.autotakeoff():

        current_lat, current_lon, current_alt, heading = c.get_current_position()
        logging.info(f"current_loc: {[current_lat,current_lon,current_alt,heading]}")
        # 将向前飞行5米（FLU坐标系）转换为全局相对高度坐标系
        new_lat, new_lon, new_alt = c.flu_to_global_relative(5, 0, 0)
        logging.info(f"new_loc: {[new_lat,new_lon,new_alt,heading]}")
        # 设置航点
        waypoints = [
            c.flu_to_global_relative(0.1, 0, 0),  # 向前飞行5米
            c.flu_to_global_relative(0.3, 0, 0),
            c.flu_to_global_relative(0.5, 0, 0),
            c.flu_to_global_relative(0.5, 0.7, 0),
            c.flu_to_global_relative(0.6, 0.8, 0),
            c.flu_to_global_relative(0.6, 1.0, 0),
            c.flu_to_global_relative(0.6, 1.2, 0),
            c.flu_to_global_relative(0.8, 1.2, 0),
            c.flu_to_global_relative(1.0, 1.2, 0),
            c.flu_to_global_relative(1.244, 1.2, 0),
            c.flu_to_global_relative(1.345, 1.2, 0),
            c.flu_to_global_relative(1.5, 1.2, 2.0),
            c.flu_to_global_relative(-1.0, 1.2, 2.0),
            c.flu_to_global_relative(-4.0, 1.2, 2.0),
            c.flu_to_global_relative(-8.0, 1.2, 2.0),
            c.flu_to_global_relative(-10.0, 1.2, 2.0),
        ]
        
        # 启动任务并监控直到完成
        c.auto_start_mission(waypoints)
        
        # 降落
        c.land()

    c.disarm()