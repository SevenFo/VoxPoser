import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
import threading

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("test_tf")
    
    tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=100))
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)
    
    node.get_logger().info("Node and TF listener initialized")
    node.create_subscription(PoseStamped,"/drone1/state/pose",lambda msg: node.get_logger().info(f"Received pose: {msg}"),rclpy.qos.qos_profile_sensor_data)
    monitor_thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    monitor_thread.start()
    while rclpy.ok():
        current_time = node.get_clock().now()
        node.get_logger().info(f"Current time: {current_time}")
        
        can_transform = tf_buffer.can_transform(
            "map",
            "camera",
            Time(),
        )
        
        if can_transform:
            node.get_logger().info("Transform available, looking up transform")
            transform = tf_buffer.lookup_transform(
                "map", "camera", Time()
            )
            node.get_logger().info(
                f"Transform from camera to map: {transform}"
            )
            break
        else:
            node.get_logger().warn(
                f"Cannot transform from camera to map, at {current_time}"
            )
        # rclpy.spin(node)
        import time
        time.sleep(1)
        # rclpy.spin_once(node, timeout_sec=1.0)
    
    node.get_logger().info("Shutting down node")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()