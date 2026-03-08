import os
import sys
import json
import httpx
import numpy as np
import base64
from datetime import datetime

# Optional ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

class DesertRoverBridge(Node if ROS2_AVAILABLE else object):
    def __init__(self):
        if not ROS2_AVAILABLE:
            print("ERROR: ROS2 (rclpy) or cv_bridge not found. This bridge requires a ROS2 environment.")
            sys.exit(1)
            
        super().__init__('desert_rover_bridge')
        
        # Parameters
        self.declare_parameter('api_url', 'http://localhost:8000/segment')
        self.api_url = self.get_parameter('api_url').get_parameter_value().string_value
        
        # Subscriptions
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
            
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/rover/navigation_status', 10)
        
        # Utils
        self.bridge = CvBridge()
        self.client = httpx.Client(timeout=10.0)
        
        self.get_logger().info('Desert Rover ROS2 Bridge Started')
        self.get_logger().info(f'Inference API: {self.api_url}')

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Encode as JPG for API
            _, buffer = cv2.imencode('.jpg', cv_image)
            
            # Send to Inference Server
            files = {'image': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
            response = self.client.post(self.api_url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                nav = result["navigation_command"]
                
                # Create ROS2 Twist message
                twist = Twist()
                
                # Logic to convert heading to angular velocity
                # heading: "forward", "bear_left", "bear_right", "stop", "emergency_stop"
                
                speed = nav["speed_kmh"] / 3.6 # Convert to m/s for ROS2
                
                if nav["stop"]:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                else:
                    twist.linear.x = speed
                    if nav["heading"] == "bear_left":
                        twist.angular.z = 0.5
                    elif nav["heading"] == "bear_right":
                        twist.angular.z = -0.5
                    else:
                        twist.angular.z = 0.0
                
                # Publish Command
                self.cmd_pub.publish(twist)
                
                # Publish Status (JSON string)
                status_msg = String()
                status_msg.data = json.dumps(nav)
                self.status_pub.publish(status_msg)
                
                self.get_logger().info(f'Published: {nav["heading"]} @ {nav["speed_kmh"]} km/h')
            else:
                self.get_logger().warn(f'API Error: {response.status_code}')
                
        except Exception as e:
            self.get_logger().error(f'Bridge Error: {e}')

def main(args=None):
    if not ROS2_AVAILABLE:
        print("ROS2 not detected. Please run in a ROS2 container or environment.")
        return

    import cv2 # Ensure cv2 is available for encoding
    
    rclpy.init(args=args)
    bridge = DesertRoverBridge()
    
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
