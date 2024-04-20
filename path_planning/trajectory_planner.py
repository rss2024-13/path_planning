import rclpy
from rclpy.node import Node
import numpy as np
import cv2

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")



    def map_cb(self, msg):
        '''
        this should discretize the grid, lmk if it doesn't
        '''
        self.kernel_thickness = 10
        
        data_from_occupancy_grid = np.reshape(msg.data, (msg.info.height, msg.info.width))
        data_from_occupancy_grid[data_from_occupancy_grid == -1] = 100  # Treat unknown as obstacles

        binary_data = np.where(data_from_occupancy_grid > 50, 0, 1).astype(np.uint8) # this makes it easier to do dilate

        # Define the structural element for the morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_thickness, self.kernel_thickness))
        thicker_lines_image = cv2.dilate(binary_data, kernel, iterations=1)

        end_result = np.where(thicker_lines_image > 0, 0, 100) #undoing the other transformation

        self.map_data = end_result



    def pose_cb(self, msg):
        # assuming we only care about x and y
        self.start_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)



    def goal_cb(self, msg):
        if self.map_data is None:
            self.get_logger().warn('No map data available.')
            return

        start = self.map_to_grid(self.start_pose)
        goal = self.map_to_grid((msg.pose.position.x, msg.pose.position.y))

        '''
        # PLAN A PATH HERE #
        path = self.planner.plan(self.map_data, start, goal) 
        self.publish_path(path)
        '''



    def map_to_grid(self, position):
        # Convert real-world coordinates to grid coordinates
        # Assuming 'position' is a tuple (x, y) in real-world coordinates
        # We need to know the resolution of the grid (meters/cell) and the origin of the map
        resolution = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y

        # Convert real-world coords to grid coords
        grid_x = int((position[0] - origin_x) / resolution)
        grid_y = int((position[1] - origin_y) / resolution)

        return (grid_x, grid_y)



    def publish_path(self, path):
        # Create a new PoseArray message
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"  # Ensure this matches the frame used in your environment

        # Convert each path point to a Pose
        resolution = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y

        for grid_pos in path:
            pose = Pose()
            # Convert grid coordinates back to real-world coordinates
            real_x = (grid_pos[0] * resolution) + origin_x
            real_y = (grid_pos[1] * resolution) + origin_y

            # Set the position
            pose.position.x = real_x
            pose.position.y = real_y
            pose.position.z = 0  # Assuming flat terrain

            # Default orientation (no rotation)
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = 0
            pose.orientation.w = 1

            # Add pose to the array
            pose_array.poses.append(pose)

        # Publish the PoseArray
        self.traj_pub.publish(pose_array)



    def plan_path(self, start_point, end_point, map):
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
