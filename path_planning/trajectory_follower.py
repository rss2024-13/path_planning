import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
 
from .utils import LineTrajectory
 
class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
 
    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")
 
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
 
        self.lookahead = 1. # m  # FILL IN #
        self.speed = 4. # m/s  # FILL IN #
        self.wheelbase_length = 0.3302 # Wheelbase, meters  # FILL IN #
 
        self.trajectory = LineTrajectory("/followed_trajectory")
 
        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic,
                                                self.pose_callback,
                                                1)
 
    def pose_callback(self, odometry_msg):
 
        # Car's current position
        x3 = odometry_msg.pose.pose.position.x
        y3 = odometry_msg.pose.pose.position.y
 
        # Loading trajectory points into an array
        points = np.array(self.trajectory.points)
 
        # x1, y1 are start of each segment, x2, y2 are end of each segment
        x1 = points[:-1, 0]
        y1 = points[:-1, 1]
        x2 = points[1:, 0]
        y2 = points[1:, 1]
 
        # Finds distances and the projection factor for each segment relative to cars position
        distances, us = self.dist_vec(x1, y1, x2, y2, x3, y3)
 
        # min_distance = np.min(distances)
 
        # Find difference between calculated distances and lookahead distance
        diff_lookahead = np.abs(distances - self.lookahead)
       
        # Index of segment that is closest to lookahead distance
        idx_min_lookahead = np.argmin(diff_lookahead)
        i = idx_min_lookahead
        u = us[i]
 
        # Finds the point on the segment that is 1 lookahead distance from car
        goal_point = (x1[i] + u * (x2[i] - x1[i]), y1[i] + u * (y2[i] - y1[i]))
 
        # Calculate lateral offset y
        y_offset = goal_point[1] - y3  # Assuming vehicle's front axle is aligned with y3
 
        # Calculate curvature kappa
        kappa = 2 * y_offset / (self.lookahead ** 2)
 
        # Calculate steering angle
        steering_angle = np.arctan(kappa * self.vehicle_wheelbase)
 
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        # drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.speed
        self.drive_pub_publish(drive_msg)
   
    def dist_vec(x1, y1, x2, y2, x3, y3):
        # x3, y3 is the car
 
        #vector from point 1 to point 2
        px = x2 - x1
        py = y2 - y1
 
        # norm squared, account for division by 0
        norm = px**2 + py**2
        norm = np.where(norm ==0, 1, norm)
 
        # u is "projection factor" of car's position onto line segment
        u = ((x3 - x1)*px + (y3 - y1)*py) / float(norm)
 
        # clip u to ensure it lies on the segment
        u = np.clip(u, 0, 1)
 
        # find nearest point on segment to car
        x = x1 + u*px
        y = y1 + u*py
 
        # vector from nearest point on segment to car
        dx = x - x3
        dy = y - y3
 
        distance = np.sqrt(dx**2 + dy**2)
 
        return distance, u
 
    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")
 
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
 
        self.initialized_traj = True
 
def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
