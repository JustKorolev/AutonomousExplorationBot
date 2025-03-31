#!/usr/bin/env python3

import curses
import numpy as np
import sys
import random
from scipy.spatial import KDTree
from .node import Node as MapNode
from .robot_controller import RobotController
import time

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA  # Fixed import for ColorRGBA

#
#   Global Definitions
#
VNOM = 0.3
WNOM = 0.5         # This gives a turning radius of v/w = 0.5m

# DIMENSIONS IN MAP RESOLUTION
RESOLUTION = 0.05
BOT_WIDTH = 8  # BOT DIMENSIONS ARE BIGGER THAN ACTUAL FOR TOLERANCE
BOT_LENGTH = 8
MAP_WIDTH = 360
MAP_HEIGHT = 240
ORIGIN_X = -9.00              # Origin = location of lower-left corner
ORIGIN_Y = -6.00

# ----------------------------------------------------
# User-configurable parameters - SEE TECHNICAL REPORT
# ----------------------------------------------------
COLLISION_CLEARANCE = 9
FRONITER_UPDATE_RADIUS = 30
COLLISION_STEP_SCALE = 9
RRT_STEP_SCALE = 3
OBSTACLE_THRESH = 80
CLEAR_THRESH = 30
CLEAR_NEIGHBOR_FRONTIER_THRESH = 5
FRONTIER_WEIGHT_DECREMENT = 0.2
FRONTIER_RADIUS = 8
HEUR_DISTANCE_WEIGHT = 5
RRT_GOAL_SELECT_FRACTION = 0.4


def wrapto180(angle):
    return angle - 2 * np.pi * round(angle / (2 * np.pi))


class CustomNode(Node):
    def __init__(self, name, vnom, wnom):
        super().__init__(name)
        self.vnom = vnom
        self.wnom = wnom

        # Publishers
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.update_odom, 1)
        self.create_subscription(OccupancyGrid, '/map', self.update_map, 1)

        # Initialize variables
        self.vel_msg = Twist()
        self.map = OccupancyGrid()
        self.map_array = np.array([[], []])
        self.frontier_weights = np.ones((MAP_HEIGHT, MAP_WIDTH), dtype=float)
        self.odom = Odometry()
        self.dt = 1 / 10.0
        self.node_start_time = self.get_clock().now().nanoseconds

    def update_map(self, msg):
        self.map = msg
        try:
            self.map_array = np.array(msg.data, dtype=np.int8).reshape((MAP_HEIGHT, MAP_WIDTH))
        except ValueError:
            self.map_array = np.array([[], []])

    def update_odom(self, msg):
        self.odom = msg

    def get_pose(self):
        pose = self.odom.pose.pose
        orientation = 2 * np.atan2(pose.orientation.z, pose.orientation.w)
        return np.array([pose.position.x, pose.position.y]), orientation

    def publish_velocity(self, lin_vel, ang_vel):
        self.vel_msg.linear.x = lin_vel
        self.vel_msg.angular.z = ang_vel
        self.pub.publish(self.vel_msg)
        self.vel_msg.linear.x = 0.0
        self.vel_msg.angular.z = 0.0

    def scale_coordinates(self, coords):
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        coords = np.array(coords, np.float64)
        coords[:, 0] -= origin_x
        coords[:, 1] -= origin_y
        return (coords / RESOLUTION).astype(int)

    def unscale_coordinates(self, coords):
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        coords = np.array(coords, np.float64)
        coords *= RESOLUTION
        coords[:, 0] += origin_x
        coords[:, 1] += origin_y
        return coords.tolist()

    def is_colliding(self, position, direction):
        step = COLLISION_STEP_SCALE * direction
        new_pos_x, new_pos_y = (self.scale_coordinates([position]) + step)[0]
        min_x = max(int(new_pos_x - COLLISION_CLEARANCE / 2), 0)
        max_x = min(int(new_pos_x + COLLISION_CLEARANCE / 2), MAP_WIDTH - 1)
        min_y = max(int(new_pos_y - COLLISION_CLEARANCE / 2), 0)
        max_y = min(int(new_pos_y + COLLISION_CLEARANCE / 2), MAP_HEIGHT - 1)

        if np.any(self.map_array[min_y:max_y + 1, min_x:max_x + 1] > OBSTACLE_THRESH):
            self.update_frontier_weights([new_pos_y, new_pos_x])
            return True
        return False

    def update_frontier_weights(self, frontier_idx):
        idx_x, idx_y = frontier_idx[1], frontier_idx[0]
        min_x = max(int(idx_x - FRONITER_UPDATE_RADIUS), 0)
        max_x = min(int(idx_x + FRONITER_UPDATE_RADIUS), MAP_WIDTH - 1)
        min_y = max(int(idx_y - FRONITER_UPDATE_RADIUS), 0)
        max_y = min(int(idx_y + FRONITER_UPDATE_RADIUS), MAP_HEIGHT - 1)
        self.frontier_weights[min_y:max_y + 1, min_x:max_x + 1] -= FRONTIER_WEIGHT_DECREMENT
        self.frontier_weights = np.maximum(self.frontier_weights, 0.01)

    def get_frontier_idxs(self):
        unexplored = ((self.map_array > CLEAR_THRESH) & (self.map_array < OBSTACLE_THRESH)).astype(int)
        unexplored_indices = np.argwhere(unexplored == 1)
        free = (self.map_array < CLEAR_THRESH).astype(int)
        blocked = (self.map_array > OBSTACLE_THRESH).astype(int)
        free_indices = np.argwhere(free == 1)
        blocked_indices = np.argwhere(blocked == 1)
        free_tree = KDTree(free_indices)
        blocked_tree = KDTree(blocked_indices)

        clear_neighbor_count = np.zeros_like(unexplored, dtype=int)
        for idx in unexplored_indices:
            neighbors = free_tree.query_ball_point(idx, FRONTIER_RADIUS)
            clear_neighbor_count[tuple(idx)] = len(neighbors)

        blocked_neighbor_count = np.zeros_like(unexplored, dtype=int)
        for idx in unexplored_indices:
            neighbors = blocked_tree.query_ball_point(idx, FRONTIER_RADIUS)
            blocked_neighbor_count[tuple(idx)] = len(neighbors)

        frontier = ((clear_neighbor_count > CLEAR_NEIGHBOR_FRONTIER_THRESH) & (blocked_neighbor_count == 0)).astype(int)
        return np.flip(np.argwhere(frontier == 1))

    def get_goal(self, robot_pos):
        frontier_idxs = self.get_frontier_idxs()
        robot_pos_x, robot_pos_y = robot_pos.x, robot_pos.y
        scaled_robot_pos = self.scale_coordinates([[robot_pos_x, robot_pos_y]])[0]

        heuristics = {}
        for idx in frontier_idxs:
            dist = np.linalg.norm(scaled_robot_pos - idx)
            tree = KDTree(frontier_idxs)
            neighbor_count = len(tree.query_ball_point(idx, FRONTIER_RADIUS))
            heuristics[tuple(idx)] = (HEUR_DISTANCE_WEIGHT * dist - neighbor_count) / self.frontier_weights[(idx[1], idx[0])]

        return min(heuristics, key=heuristics.get) if heuristics else None

    def publish_frontier_markers(self, points):
        unscaled_points = self.unscale_coordinates(points)
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.ns = "frontier_markers"
        marker.scale.x = 0.1
        marker.scale.y = 0.1

        for p_unscaled, p in zip(unscaled_points, points):
            point = Point(x=float(p_unscaled[0]), y=float(p_unscaled[1]))
            marker.points.append(point)

            color = ColorRGBA(
                r=1 - self.frontier_weights[p[1], p[0]] * 0.9,
                g=self.frontier_weights[p[1], p[0]],
                b=0.0,
                a=1.0
            )
            marker.colors.append(color)

        self.marker_pub.publish(marker)

    def publish_goal_marker(self, point):
        unscaled_point = self.unscale_coordinates([point])[0]
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 1
        marker.ns = "goal_marker"

        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.color.r = 0.0
        marker.color.g = 0.2
        marker.color.b = 1.0
        marker.color.a = 1.0

        goal_point = Point()
        goal_point.x, goal_point.y = float(unscaled_point[0]), float(unscaled_point[1])
        marker.points.append(goal_point)

        self.marker_pub.publish(marker)

    def publish_path_marker(self, points):
        unscaled_points = self.unscale_coordinates(points)
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 2
        marker.ns = "path"

        for pair in unscaled_points:
            p1 = Point()
            p1.x, p1.y = float(pair[0]), float(pair[1])
            marker.points.append(p1)

        marker.scale.x = 0.05  # Line width
        marker.color.r = 0.0
        marker.color.g = 0.7
        marker.color.b = 0.5
        marker.color.a = 1.0

        self.marker_pub.publish(marker)

    # Shutdown
    def shutdown(self):
        # Destroy the rate and node.
        self.destroy_rate(self.rate)
        self.destroy_node()


    def rrt(self, startnode, goalnode):
        SMAX = 25000
        NMAX = 10000

        # Start the tree with the startnode (set no parent just in case).
        startnode.parent = None
        tree = [startnode]

        # Function to attach a new node to an existing node: attach the
        # parent, add to the tree, and show in the figure.
        def addtotree(oldnode, newnode):
            newnode.parent = oldnode
            tree.append(newnode)

        # Loop - keep growing the tree.
        steps = 0
        while True:
            # Determine the target state.
            if random.random() <= RRT_GOAL_SELECT_FRACTION:
                target_coords = np.array(goalnode.coordinates())
            else:
                target_coords = np.array([random.randint(0, MAP_WIDTH-1),
                                        random.randint(0, MAP_HEIGHT-1)])
            targetnode = MapNode(*target_coords)

            # Directly determine the distances to the target node.
            distances = np.array([node.distance(targetnode) for node in tree])
            index     = np.argmin(distances)
            nearnode  = tree[index]
            d         = distances[index]

            # Determine the next node.
            near_coords = np.array(nearnode.coordinates())
            if np.array_equal(target_coords, near_coords):
                continue
            step_size = RRT_STEP_SCALE / d
            nextnode_coords = (near_coords + (target_coords - near_coords) * step_size).astype(int)

            if np.array_equal(near_coords, nextnode_coords):
                print("COORDINATAES ARE THE SAME - THIS SHOULD NEVER HAPPEN")


            nextnode = MapNode(*nextnode_coords)


            # Check whether to attach.
            if self.map_array.shape == (MAP_HEIGHT, MAP_WIDTH):
                if nextnode.inFreespace(self.map_array) and nearnode.connectsTo(nextnode, self.map_array):
                    addtotree(nearnode, nextnode)

                    # If within DSTEP, also try connecting to the goal.  If
                    # the connection is made, break the loop to stop growing.
                    if nextnode.distance(goalnode) <= RRT_STEP_SCALE:
                        addtotree(nextnode, goalnode)
                        break

            else:
                continue

            # Check whether we should abort - too many steps or nodes.
            steps += 1
            if (steps >= SMAX) or (len(tree) >= NMAX):
                self.update_frontier_weights(goalnode.coordinates())
                print("Aborted after %d steps and the tree having %d nodes" %
                    (steps, len(tree)))
                return None

        # Build the path.
        path = [goalnode]
        while path[0].parent is not None:
            path.insert(0, path[0].parent)
        return path


    # Post process the path.
    def PostProcess(self, path):
        if len(path) < 3:
            return

        new_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if (path[j]).connectsTo(path[i], self.map_array):
                    break
                j -= 1

            new_path.append(path[j])
            i = j

        return new_path

    def loop(self, screen):
        controller = RobotController()

        while rclpy.ok():
            rclpy.spin_once(self)
            now = self.get_clock().now().nanoseconds
            if now - self.node_start_time < 5e9:
                continue

            pose = self.odom.pose.pose
            pos = pose.position
            goal = self.get_goal(pos)

            # Quit if no goal (nothing left to explore)
            if goal is None:
                print("Exploration Finished! Killing program.")
                break

            # Visualize frontiers
            frontier_idxs = self.get_frontier_idxs()
            self.publish_frontier_markers(frontier_idxs)

            screen.addstr(9, 0, f"Goal: {goal}")
            self.publish_goal_marker(goal)

            # Calculate path using RRT
            pos_x, pos_y = pos.x, pos.y
            orientation = 2 * np.atan2(pose.orientation.z, pose.orientation.w)
            scaled_pos_x, scaled_pos_y = self.scale_coordinates([(pos_x, pos_y)])[0]
            screen.addstr(10, 0, f"position: {scaled_pos_x, scaled_pos_y}")
            screen.addstr(11, 0, f"orientation: {orientation}")

            path = self.rrt(MapNode(scaled_pos_x, scaled_pos_y), MapNode(*goal))
            if path:
                path = self.PostProcess(path)
                path_points = [node.coordinates() for node in path]
                unscaled_path_points = self.unscale_coordinates(path_points)
                unscaled_path_points[0] = (pos_x, pos_y)
                self.publish_path_marker(path_points)
                controller.update_path(unscaled_path_points)

                controller.run(self, self.get_pose, self.publish_velocity,
                                self.is_colliding, rclpy.spin_once)
                controller.reset()

            screen.refresh()

    def shutdown(self):
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CustomNode('auton', VNOM, WNOM)
    try:
        curses.wrapper(node.loop)
    except KeyboardInterrupt:
        pass
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
