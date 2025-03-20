#!/usr/bin/env python3
#
#   auton.py
#
#   Continually (at 10Hz!) send the a velocity command.
#
#   Node:       /auton
#
#   Subscribe:  map, odom
#   Publish:    /cmd_vel        geometry_msgs/Twist
#
import curses
import numpy as np
import sys
import random
from scipy.spatial      import KDTree
from .node import Node as MapNode
from .robot_controller import RobotController
import time

# ROS Imports
import rclpy

from rclpy.node                 import Node
from rclpy.time                 import Time

from rclpy.qos                  import QoSProfile, DurabilityPolicy
from geometry_msgs.msg          import Twist
from std_msgs.msg import ColorRGBA
from nav_msgs.msg               import Odometry
from nav_msgs.msg               import OccupancyGrid
from visualization_msgs.msg     import Marker, MarkerArray
from geometry_msgs.msg import Point

#
#   Global Definitions
#
VNOM = 0.3
WNOM = 0.5         # This gives a turning radius of v/w = 0.5m

# DIMENSTION IN MAP RESOLUTION
RESOLUTION = 0.05
BOT_WIDTH = 8 # BOT DIMENSIONS ARE BIGGER THAN ACTUAL FOR TOLERANCE
BOT_LENGTH = 8
MAP_WIDTH  = 360
MAP_HEIGHT = 240
ORIGIN_X   = -9.00              # Origin = location of lower-left corner
ORIGIN_Y   = -6.00

# TUNABLE CONSTANTS
COLLISION_CLEARANCE = 9
FRONITER_UPDATE_RADIUS = 10
COLLISION_STEP_SCALE = 9
RRT_STEP_SCALE = 4
OBSTACLE_THRESH = 80
CLEAR_THRESH = 30
CLEAR_NEIGHBOR_FRONTIER_THRESH = 5
FRONTIER_WEIGHT_DECREMENT = 0.2
FRONTIER_RADIUS = 8
HEUR_DISTANCE_WEIGHT = 5
RRT_GOAL_SELECT_FRACTION = 0.3

def wrapto180(angle):
    return angle - 2*np.pi * round(angle/(2*np.pi))

#
#   Custom Node Class
#
class CustomNode(Node):
    # Initialization.
    def __init__(self, name, vnom, wnom):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Save the parameters.
        self.vnom = vnom
        self.wnom = wnom

        # Create a publisher to send twist commands.
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create a publisher to send marker arrays
        self.marker_pub = self.create_publisher(Marker,
                                        '/visualization_marker', 10)

        # Subscribers
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.create_subscription(Odometry, '/odom', self.updateOdom, 1)
        self.create_subscription(OccupancyGrid, '/map', self.updateMap, 1)


        # Initialize the (repeating) message data.
        self.vel_msg = Twist()
        self.vel_msg.linear.x = 0.0
        self.vel_msg.linear.y = 0.0
        self.vel_msg.linear.z = 0.0
        self.vel_msg.angular.x = 0.0
        self.vel_msg.angular.y = 0.0
        self.vel_msg.angular.z = 0.0

        # Initialize map and odometry
        self.map = OccupancyGrid()
        self.map_array = np.array([[], []])
        self.frontier_weights = np.ones(shape=(MAP_HEIGHT, MAP_WIDTH), dtype=float)
        self.odom = Odometry()

        # Create a fixed rate to control the speed of sending commands.
        rate    = 10.0
        self.dt = 1/rate
        self.rate = self.create_rate(rate)
        self.node_start_time = self.get_clock().now().nanoseconds

        # Report.
        self.get_logger().info("Auton sending every %f sec..." % self.dt)
        self.get_logger().info("Nominal fwd = %6.3fm/s, spin = %6.3frad/sec"
                               % (self.vnom, self.wnom))

    def updateMap(self, msg):
        self.map = msg
        try:
            self.map_array = np.array(msg.data, dtype=np.int8).reshape(
                                                        (MAP_HEIGHT, MAP_WIDTH))
        except ValueError:
            self.map_array = np.array([[], []])

    def updateOdom(self, msg):
        self.odom = msg

    def get_pose(self):
        pose = self.odom.pose.pose
        orientation = 2 * np.atan2(pose.orientation.z,
                                   pose.orientation.w)
        return np.array([pose.position.x, pose.position.y]),\
               orientation

    def publish_velocity(self, lin_vel, ang_vel):
        self.vel_msg.linear.x  = lin_vel
        self.vel_msg.angular.z = ang_vel
        self.pub.publish(self.vel_msg)

        # Reset velocities?
        self.vel_msg.linear.x  = 0.0
        self.vel_msg.angular.z = 0.0

    def unscale_coordinates(self, coords):
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        coords = np.array(coords, np.float64)
        unscaled_coords = coords * RESOLUTION
        unscaled_coords[:,1] += origin_x
        unscaled_coords[:,0] += origin_y
        return list(unscaled_coords)

    def scale_coordinates(self, coords):
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        coords = np.array(coords, np.float64)
        coords[:,1] -= origin_x
        coords[:,0] -= origin_y
        scaled_coords = coords / RESOLUTION
        return list(scaled_coords.astype(int))


    def is_colliding(self, position, direction):
        step = COLLISION_STEP_SCALE * np.flip(direction)
        pos_x, pos_y = position
        new_pos_y, new_pos_x = (np.array(self.scale_coordinates([[pos_y, pos_x]])) + step)[0]
        min_x = max(int(new_pos_x - COLLISION_CLEARANCE / 2), 0)
        max_x = min(int(new_pos_x + COLLISION_CLEARANCE / 2), MAP_WIDTH - 1)
        min_y = max(int(new_pos_y - COLLISION_CLEARANCE / 2), 0)
        max_y = min(int(new_pos_y + COLLISION_CLEARANCE / 2), MAP_HEIGHT - 1)

        # self.publish_goal_marker([min_y, min_x])
        # self.publish_goal_marker([max_y, max_x])

        if np.any(self.map_array[min_y:max_y+1, min_x:max_x+1] > OBSTACLE_THRESH):
            self.update_frontier_weights([new_pos_y, new_pos_x])
            print("COLLISION DETECTED AHEAD")
            return True  # Collision detected
        return False  # No collision


    def update_frontier_weights(self, frontier_idx):
        idx_x = frontier_idx[1]
        idx_y = frontier_idx[0]
        min_x = max(int(idx_x - FRONITER_UPDATE_RADIUS), 0)
        max_x = min(int(idx_x + FRONITER_UPDATE_RADIUS), MAP_WIDTH - 1)
        min_y = max(int(idx_y - FRONITER_UPDATE_RADIUS), 0)
        max_y = min(int(idx_y + FRONITER_UPDATE_RADIUS), MAP_HEIGHT - 1)

        self.frontier_weights[min_y:max_y+1, min_x:max_x+1] -= FRONTIER_WEIGHT_DECREMENT
        self.frontier_weights = np.maximum(self.frontier_weights, 0.01)



    def get_frontier_idxs(self):
        # FIND FRONTIER CELLS
        unexplored = ((self.map_array > CLEAR_THRESH) & (self.map_array < OBSTACLE_THRESH)).astype(int)
        unexplored_indices = np.argwhere(unexplored == 1)
        free = (self.map_array < CLEAR_THRESH).astype(int)
        blocked = (self.map_array > OBSTACLE_THRESH).astype(int)
        free_indices = np.argwhere(free == 1)
        blocked_indices = np.argwhere(blocked == 1)
        free_tree = KDTree(free_indices)
        blocked_tree = KDTree(blocked_indices)

        clear_neighbor_count = np.zeros_like(unexplored, dtype=int) # num of clear neighbors
        for idx in unexplored_indices:
            neighbors = free_tree.query_ball_point(idx, FRONTIER_RADIUS)
            clear_neighbor_count[tuple(idx)] = len(neighbors)

        blocked_neighbor_count = np.zeros_like(unexplored, dtype=int) # num of blocked neighbors
        for idx in unexplored_indices:
            neighbors = blocked_tree.query_ball_point(idx, FRONTIER_RADIUS)
            blocked_neighbor_count[tuple(idx)] = len(neighbors)

        frontier = ((clear_neighbor_count > CLEAR_NEIGHBOR_FRONTIER_THRESH) & (blocked_neighbor_count == 0)).astype(int)
        frontier_indices = np.argwhere(frontier == 1)

        return frontier_indices

    def get_goal(self, robot_pos):
        frontier_idxs = self.get_frontier_idxs()
        self.publish_frontier_markers(frontier_idxs)
        robot_pos_x = robot_pos.x
        robot_pos_y = robot_pos.y
        scaled_robot_pos = self.scale_coordinates([[robot_pos_y, robot_pos_x]])[0]


        heuristics = {} # heuristic for each frontier cell
        for idx in frontier_idxs:
            # make sure this distance is correct
            dist = np.sqrt((scaled_robot_pos[1] - idx[1])**2 + (scaled_robot_pos[0] - idx[0])**2)

            # include number of neighboring frontier cells in cost
            tree = KDTree(frontier_idxs)
            neighbor_count = len(tree.query_ball_point(idx, FRONTIER_RADIUS)) # num of frontier neighbors

            heuristics[tuple(idx)] = (HEUR_DISTANCE_WEIGHT*dist - neighbor_count) / self.frontier_weights[tuple(idx)]

        # pick goal with minimum cost
        if len(heuristics) != 0:
            goal_idx = min(heuristics, key=heuristics.get)
        else:
            goal_idx = None
        return goal_idx

    def publish_frontier_markers(self, points):
        # Markers for frontier points
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

        for (i, p_unscaled), p  in zip(enumerate(unscaled_points), points):
            point = Point()
            point.x, point.y = float(p_unscaled[1]), float(p_unscaled[0])
            marker.points.append(point)

            color = ColorRGBA()
            color.r = 1 - self.frontier_weights[p[0], p[1]] * 0.9
            color.g = self.frontier_weights[p[0], p[1]]
            color.b = 0.0
            color.a = 1.0  # Full opacity
            marker.colors.append(color)

        self.marker_pub.publish(marker)

    def publish_goal_marker(self, point):
        # marker for goal
        unscaled_point = self.unscale_coordinates([point])[0]
        marker1 = Marker()
        marker1.header.frame_id = "map"
        marker1.type = Marker.POINTS
        marker1.action = Marker.ADD
        marker1.header.stamp = self.get_clock().now().to_msg()
        # marker1.lifetime = rclpy.duration.Duration(seconds=4).to_msg()
        marker1.id = 1
        marker1.ns = "goal_marker"

        marker1.scale.x = 0.3
        marker1.scale.y = 0.3
        marker1.color.r = 0.0
        marker1.color.g = 0.2
        marker1.color.b = 1.0
        marker1.color.a = 1.0

        goal_point = Point()
        goal_point.x, goal_point.y = float(unscaled_point[1]), float(unscaled_point[0])
        marker1.points.append(goal_point)

        self.marker_pub.publish(marker1)


    def publish_path_marker(self, points):
        unscaled_points = self.unscale_coordinates((points))
        marker = Marker()
        marker.header.frame_id = "map"  # Change "map" to "odom" if "map" doesn't exist
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.header.stamp = self.get_clock().now().to_msg()
        # marker.lifetime = rclpy.duration.Duration(seconds=2).to_msg()
        marker.id = 1
        marker.ns = "path"
        for pair in unscaled_points:
            p1 = Point()
            p1.x, p1.y = float(pair[1]), float(pair[0])

            marker.points.append(p1)

            marker.scale.x = 0.05 # Line width
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
                target_coords = np.array([random.randint(0, MAP_HEIGHT-1),
                                        random.randint(0, MAP_WIDTH-1)])
            targetnode = MapNode(*target_coords)

            # Directly determine the distances to the target node.
            distances = np.array([node.distance(targetnode) for node in tree])
            index     = np.argmin(distances)
            nearnode  = tree[index]

            # Determine the next node.
            near_coords = np.array(nearnode.coordinates())
            if np.array_equal(target_coords, near_coords):
                continue

            step = RRT_STEP_SCALE * (target_coords - near_coords) / np.linalg.norm((target_coords - near_coords))
            nextnode_coords = (np.array(near_coords + step)).astype(int)

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

        # Report and return.
        # print("Finished after %d steps and the tree having %d nodes" %
        #     (steps, len(tree)))
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

    # def PostProcess(self, path):
    #     i = 0
    #     while (i < len(path)-2):
    #         if path[i].connectsTo(path[i+2], self.map_array):
    #             path.pop(i+1)
    #         else:
    #             i += 1
    #     return path


    # Run the terminal input loop, send the commands, and ROS spin.
    def loop(self, screen):
        # Initialize the controller
        controller = RobotController()


        # Run the loop until shutdown.
        while rclpy.ok():

            # wait to receive correct data
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

            screen.addstr(9, 0, f"Goal: {goal}")
            self.publish_goal_marker(goal)


            # Calculate path using RRT
            pos_x, pos_y = pos.x, pos.y
            orientation = 2 * np.atan2(pose.orientation.z,
                                       pose.orientation.w)
            scaled_pos_y, scaled_pos_x = self.scale_coordinates([(pos_y, pos_x)])[0]
            screen.addstr(10, 0, f"position: {scaled_pos_x, scaled_pos_y}")
            screen.addstr(11, 0, f"orientation: {orientation}")

            if scaled_pos_x != goal[0] and scaled_pos_y != goal[1]:
                path = self.rrt(MapNode(scaled_pos_y, scaled_pos_x), MapNode(*goal))
                if not path == None:
                    path = self.PostProcess(path)
                    path_points = [node.coordinates() for node in path]
                    unscaled_path_points = self.unscale_coordinates(path_points)
                    unscaled_path_points[0] = (pos_y, pos_x)
                    self.publish_path_marker(path_points)
                    controller.update_path(unscaled_path_points)


                    controller.run(self, self.get_pose, self.publish_velocity,
                                   self.is_colliding, rclpy.spin_once)
                    controller.reset()


                else:
                    print("NO PATH FOUND")
            # except Exception as e:
            #     print("failed")
            #     print(e)




            screen.clrtoeol()
            screen.refresh()

            # Spin once to process other items.





#
#   Main Code
#
def main(args=None):
    # Pull out the nominal forward/spin speeds from the non-ROS parameters.
    if (len(sys.argv) > 3):
        print("Usage: auton.py forward_speed spin_speed")
        print("GOOD DEFAULTS: auton.py %3.1f %3.1f" % (VNOM, WNOM))
        return
    elif (len(sys.argv) < 3):
        print("Usage: auton.py forward_speed spin_speed")
        print("Using default values...")
        vnom = VNOM
        wnom = WNOM
    else:
        vnom = float(sys.argv[1])
        wnom = float(sys.argv[2])

    # Initialize ROS.
    rclpy.init(args=args)

    # Instantiate the node.
    node = CustomNode('auton', vnom, wnom)

    # Run the terminal input loop, which spins the ROS items.
    try:
        curses.wrapper(node.loop)
    except KeyboardInterrupt:
        pass

    # Shutdown the node and ROS.
    node.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
