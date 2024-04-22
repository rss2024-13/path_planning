import rclpy
from rclpy.node import Node
import numpy as np
import cv2
# import networkx as nx

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
import heapq


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

        self.map_data = None
        self.graph = None
        self.map = None


    def map_cb(self, msg):
        '''
        This is called to make the discretized map and graph based on data that comes in.
        '''
        self.map = msg
        self.kernel_thickness = 30
        
        data_from_occupancy_grid = np.reshape(msg.data, (msg.info.height, msg.info.width))
        data_from_occupancy_grid[data_from_occupancy_grid == -1] = 100  # Treat unknown as obstacles

        binary_data = np.where(data_from_occupancy_grid > 50, 1, 0).astype(np.uint8) # this makes it easier to do dilate

        # Define the structural element for the morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_thickness, self.kernel_thickness))
        thicker_lines_image = cv2.dilate(binary_data, kernel, iterations=1)

        end_result = np.where(thicker_lines_image > 0, 100, 0) #undoing the other transformation

        self.map_data = end_result
        # self.graph = self.create_graph(self.map_data, True)
        self.get_logger().info('finished making graph')
        
    def astar(self, start, goal, neighbors, heuristic):
        open_list = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return None


    def pose_cb(self, msg):
        self.get_logger().info('got start.')
        # assuming we only care about x and y
        self.start_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)



    def goal_cb(self, msg):
        self.get_logger().info('got path.')
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
        def dist(a, b):
            (x1, y1) = a
            (x2, y2) = b
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        
        include_diagonals = True
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        if include_diagonals:
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # Diagonals

        def neighbors(node):
            neighbors = []
            for direction in directions:
                x_change, y_change = direction
                new_node = (node[0] + x_change, node[1] + y_change)
                if self.map_data[new_node[1], new_node[0]] == 0:
                    neighbors.append(new_node)
            return neighbors
        
        path = self.astar(start, goal, neighbors, dist)
        if path is not None:
            self.get_logger().info('found path.')
            self.publish_path(path)
        else:
            self.get_logger().info('path not found')


    

    # def create_graph(self, image, include_diagonals=False):
    #     """
    #     Create a graph from an image where each white pixel is connected to its
    #     neighboring white pixels. Optionally include diagonal connections.
    #     """
    #     # Ensure the image is in binary format (i.e., only contains 0s and 255s)
    #     # image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

    #     # Initialize the graph
    #     G = nx.Graph()
        
    #     # Get the dimensions of the image
    #     rows, cols = image.shape
    #     self.get_logger().info('rows')
    #     self.get_logger().info(str(rows))
    #     self.get_logger().info('cols')
    #     self.get_logger().info(str(cols))
        
    #     # Define the directions for neighbors (8 directions if diagonals included, else 4)
    #     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    #     if include_diagonals:
    #         directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])  # Diagonals
        
    #     # Iterate over the image pixels
    #     for y in range(rows):
    #         for x in range(cols):
    #             # Check if the current pixel is not an obstacle
    #             if image[y, x] == 0:
    #                 # Add the current pixel as a node to the graph
    #                 G.add_node((x, y))
    #                 # Check for white neighbors
    #                 for dx, dy in directions:
    #                     neighbor_x, neighbor_y = x + dx, y + dy
    #                     # Check if the neighbor coordinates are inside the image bounds
    #                     if 0 <= neighbor_y < rows and 0 <= neighbor_x < cols:
    #                         # Check if the neighbor is not an obstacle
    #                         if image[neighbor_y, neighbor_x] == 0:
    #                             # Add the neighbor as a node and connect it to the current pixel
    #                             G.add_node((neighbor_x, neighbor_y))
    #                             G.add_edge((x, y), (neighbor_x, neighbor_y))
    #     return G



    def map_to_grid(self, position):
        # Convert real-world coordinates to grid coordinates
        # Assuming 'position' is a tuple (x, y) in real-world coordinates
        # We need to know the resolution of the grid (meters/cell) and the origin of the map
        resolution = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        orientation = self.map.info.origin.orientation 

        real_coords = np.array([position[0], position[1]]) - np.array([origin_x, origin_y])
        real_coords = np.append(real_coords, 0)
        rotation_matrix = self.quaternion_to_rotation_matrix(orientation)
        inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
        pixel_coords = np.dot(inverse_rotation_matrix, real_coords)

        u, v = pixel_coords[:2] / resolution

        return (int(u), int(v))
    
    def quaternion_to_rotation_matrix(self, quaternion):
        x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])



    def publish_path(self, path):
        self.trajectory.clear()
        # Convert each path point to a Pose
        resolution = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        orientation = self.map.info.origin.orientation 
        rotation_matrix = self.quaternion_to_rotation_matrix(orientation)

        for grid_pos in path:
            # Convert grid coordinates back to real-world coordinates
            pixel_coords = np.array([grid_pos[0], grid_pos[1]]) * resolution
            pixel_coords = np.append(pixel_coords, 0)
            real_coords = np.dot(rotation_matrix, pixel_coords)
            real_coords = real_coords[:2]
            real_coords += np.array([origin_x, origin_y])

            self.trajectory.addPoint((real_coords[0], real_coords[1]))

        # Publish the PoseArray
        self.plan_path()



    def plan_path(self):
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
