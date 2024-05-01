import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped, PoseArray, Point

from .utils import LineTrajectory
from typing import List, Tuple

import math
import numpy as np


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = "/path_planner_drive" # self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 1  # FILL IN #
        self.speed = 1.0  # FILL IN #
        self.wheelbase_length = 0.3302  # FILL IN #
        self.lfw_ratio = 1/4
        self.turn_radius = .1
        self.max_steering_ang = .34
        self.curr_steering_ang = 0.0

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.pp_pub = self.create_publisher(Marker, "/pp_marker", 1)
        self.eta_pub = self.create_publisher(Marker, "/eta_marker", 1)
        self.circle_pub = self.create_publisher(Marker, "/circle_marker", 1)
        
        self.traj_poses = None
        self.initialized_traj = False
        self.curr_pose = None

    def pose_callback(self, odometry_msg):
        pose = odometry_msg.pose.pose  # Extract the Pose from the PoseWithCovarianceStamped message
        x, y, theta = self.pose_to_xyt(pose) # world frame
        self.curr_pose = (x, y, theta)

        drive_cmd = AckermannDriveStamped()

        # self.get_logger().info(f"Current Pose [{x}, {y}, {theta}]")

        if self.initialized_traj:
            traj_points_x = np.array([pose.position.x for pose in self.traj_poses])
            traj_points_y = np.array([pose.position.y for pose in self.traj_poses])

            point_dist = ((x - traj_points_x) ** 2 + (y - traj_points_y) ** 2) ** .5
            idx = np.argmin(point_dist)
            curr_point = (traj_points_x[idx], traj_points_y[idx])

            if idx >= len(traj_points_x)-1: # at end point
                drive_cmd.drive.speed = 0.0
                self.curr_steering_ang = self.curr_steering_ang
                self.initialized_traj = False
            else:
                i = idx+1
                d = ((curr_point[0] - traj_points_x[i]) ** 2 + (curr_point[1] - traj_points_y[i]) ** 2) ** .5
                while d < self.lookahead and i < len(traj_points_x)-1:
                    i += 1
                    d = ((curr_point[0] - traj_points_x[i]) ** 2 + (curr_point[1] - traj_points_y[i]) ** 2) ** .5
                
                pp = (traj_points_x[i], traj_points_y[i]) # pursuit point

                self.mark_pt(self.pp_pub, (0.0, 1.0, 0.0), [(pp[0], pp[1])])

                # angle_to_cone = np.arctan2(pp[1] - y, pp[0] - x)

                theta_pp = math.atan2(pp[1]-x, pp[0]-y)
                angle_sign = np.sign(-math.sin(theta) * (pp[0]-x) + math.cos(theta) *( pp[1] - y) )

                dot = (pp[0] - x) * (math.cos(theta)) + (pp[1] - y) * (math.sin(theta))
                mag_curr_pos = 1 #(x ** 2 + y ** 2) ** .5
                mag_pp_pos = ((pp[0] - x) ** 2 + (pp[1] - y) ** 2) ** .5

                eta = angle_sign * math.acos(dot / (mag_curr_pos * mag_pp_pos))
                

                self.mark_pt(self.eta_pub, (0.0, 1.0, 1.0), [(x, y), (x+math.cos(theta+eta), y+math.sin(theta+eta))])

                l_1 = ((x - pp[0]) ** 2 + (y - pp[1]) ** 2) ** .5
                L = self.wheelbase_length
                self.turn_radius = l_1 / (2 * math.sin(eta))
                Lfw = (self.turn_radius ** 2 + self.wheelbase_length ** 2) ** .5
                lfw = self.wheelbase_length * self.lfw_ratio

                R = l_1/(2*math.sin(eta))
                ccx = x +  R * math.cos(theta- math.pi/2) # circle center x
                ccy = y +  R * math.sin(theta- math.pi/2)
                self.mark_pt(self.circle_pub, (1.0, 1.0, 0.0), [(ccx + R*math.cos(t), ccy+R*math.sin(t)) for t in np.linspace(0,2*math.pi, 32)])


                cmd = math.atan2((2 * self.wheelbase_length * math.sin(eta)), l_1)
                # cmd = -1 * math.atan2(L * math.sin(eta),  (Lfw/2 + lfw * math.cos(eta)))
                # self.get_logger().info(f"eta: {eta}")
                # self.get_logger().info(f"CMD: {cmd}")
                # self.get_logger().info(f"sign of sin(theta_pp - theta): {np.sign(math.sin(theta_pp - theta))} , sign of y_pp in robot frame: {np.sign(-math.sin(theta) * (pp[0]-x) + math.cos(theta) *( pp[1] - y) )}")

                drive_cmd.drive.steering_angle = angle_sign * min(abs(cmd), self.max_steering_ang)

                drive_cmd.drive.speed = self.speed
                self.curr_steering_ang = drive_cmd.drive.steering_angle
        else:
            drive_cmd.drive.speed = 0.0
            self.curr_steering_ang = self.curr_steering_ang

        self.drive_pub.publish(drive_cmd)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True

        self.traj_poses = msg.poses

    def pose_to_xyt(self, pose):
        '''
        Convert particles between representations of position/orientation
         
        args:
            array of Pose (ROS data structure that contains position as (x,y,z)
            and orientation as (x,y,z,w))
        returns:
            array of [x,y,t] structure where x = position along x, y = position
            along y, and t = theta along xy-plane
        '''
        x = pose.position.x
        y = pose.position.y
        
        # calculating yaw from quaternion (theta)
        t3 = +2.0 * (pose.orientation.w * pose.orientation.z + pose.orientation.x * pose.orientation.y)
        t4 = +1.0 - 2.0 * (pose.orientation.y * pose.orientation.y + pose.orientation.z * pose.orientation.z)
        theta = np.arctan2(t3, t4)
        # done with theta calculation

        return np.array([x, y, theta])
    
    def tuple_to_point(self, data_points: List[Tuple[float, float]]) -> List[Point]:
        return [Point(x=x, y=y) for x, y in data_points]
    
    def mark_pt(self, subscriber, color_tup, data):
        msg_data = self.tuple_to_point(data)

        mark_pt = Marker()
        mark_pt.header.frame_id = "/map"
        mark_pt.header.stamp = self.get_clock().now().to_msg()
        mark_pt.type = mark_pt.SPHERE_LIST
        mark_pt.action = mark_pt.ADD
        mark_pt.scale.x = .5
        mark_pt.scale.y = .5
        mark_pt.scale.z = .5
        mark_pt.color.a = 1.0
        mark_pt.color.r = color_tup[0]
        mark_pt.color.g = color_tup[1]
        mark_pt.color.b = color_tup[2]
        mark_pt.points = msg_data
        subscriber.publish(mark_pt)


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
