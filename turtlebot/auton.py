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

# ROS Imports
import rclpy

from rclpy.node                 import Node
from rclpy.time                 import Time

from rclpy.qos                  import QoSProfile, DurabilityPolicy
from geometry_msgs.msg          import Twist
from nav_msgs.msg               import Odometry
from nav_msgs.msg               import OccupancyGrid
from visualization_msgs.msg     import Marker
from geometry_msgs.msg import Point

#
#   Global Definitions
#
VNOM = 0.3
WNOM = 0.5         # This gives a turning radius of v/w = 0.5m

# DIMENSTION IN MAP RESOLUTION
RESOLUTION = 0.05
BOT_WIDTH = 7 # BOT DIMENSIONS ARE BIGGER THAN ACTUAL FOR TOLERANCE
BOT_LENGTH = 7
MAP_WIDTH  = 360
MAP_HEIGHT = 240
ORIGIN_X   = -9.00              # Origin = location of lower-left corner
ORIGIN_Y   = -6.00

OBSTACLE_THRESH = 80
CLEAR_THRESH = 30
FRONTIER_DIST = 1

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
        self.msg = Twist()
        self.msg.linear.x = 0.0
        self.msg.linear.y = 0.0
        self.msg.linear.z = 0.0
        self.msg.angular.x = 0.0
        self.msg.angular.y = 0.0
        self.msg.angular.z = 0.0

        # Initialize map and odometry
        self.map = OccupancyGrid()
        self.map_array = np.array([[], []])
        self.odom = Odometry()

        # Create a fixed rate to control the speed of sending commands.
        rate    = 10.0
        self.dt = 1/rate
        self.rate = self.create_rate(rate)
        self.timer = self.create_timer(1.0, self.publish_markers)

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

    def is_colliding(self, pos, heading, forward_vel_multiplier,
                     angular_vel_multiplier, screen):
        pos_x, pos_y = pos.x, pos.y
        pos_x_scaled, pos_y_scaled = (pos_x - ORIGIN_X) / RESOLUTION, (pos_y - ORIGIN_Y) / RESOLUTION
        dstep = 4 * np.sign(forward_vel_multiplier)
        dtheta = 0.1 * np.sign(angular_vel_multiplier)
        next_pos_x = pos_x_scaled + dstep * np.cos(heading + dtheta/2) * np.sinc(dtheta/2/np.pi)
        next_pos_y = pos_y_scaled + dstep * np.sin(heading + dtheta/2) * np.sinc(dtheta/2/np.pi)
        screen.addstr(12, 0,
                        f"Scaled pos: {next_pos_x, next_pos_y}")
        min_x = max(int(next_pos_x - BOT_WIDTH / 2), 0)
        max_x = min(int(next_pos_x + BOT_WIDTH / 2), MAP_WIDTH - 1)
        min_y = max(int(next_pos_y - BOT_LENGTH / 2), 0)
        max_y = min(int(next_pos_y + BOT_LENGTH / 2), MAP_HEIGHT - 1)

        # screen.addstr(16, 0, f"Map stuff: {np.where(map_array > OBSTACLE_THRES)[0]}")
        if self.map_array.shape == (MAP_HEIGHT, MAP_WIDTH) and\
            np.any(self.map_array[min_y:max_y+1, min_x:max_x+1] > OBSTACLE_THRESH):
            return True  # Collision detected
        return False  # No collision

    def unscale_coordinates(self, coords):
        RESOLUTION = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        coords = np.array(coords)
        unscaled_coords = coords * RESOLUTION
        unscaled_coords[:,1] += origin_x
        unscaled_coords[:,0] += origin_y
        return list(unscaled_coords)

    def get_frontier_idxs(self):
        # FIND FRONTIER CELLS
        unexplored = ((self.map_array > CLEAR_THRESH) & (self.map_array < OBSTACLE_THRESH)).astype(int)
        unexplored_indices = np.argwhere(unexplored == 1)
        free = (self.map_array < CLEAR_THRESH).astype(int)
        free_indices = np.argwhere(free == 1)
        tree = KDTree(free_indices)

        neighbor_count = np.zeros_like(unexplored, dtype=int) # num of clear neighbors
        for idx in unexplored_indices:
            neighbors = tree.query_ball_point(idx, FRONTIER_DIST)
            neighbor_count[tuple(idx)] = len(neighbors)

        frontier = ((neighbor_count > 0)).astype(int)
        frontier_indices = np.argwhere(frontier == 1)

        return frontier_indices

    def get_goal(self, pos, frontier_idxs):
        heuristics = {} # heuristic for each frontier cell
        for idx in frontier_idxs:
            # make sure this distance is correct
            dist = np.sqrt((pos.x - idx[0])**2 + (pos.y - idx[1])**2)

            # include number of neighboring frontier cells in cost
            tree = KDTree(frontier_idxs)
            neighbor_count = len(tree.query_ball_point(idx, FRONTIER_DIST)) # num of frontier neighbors

            heuristics[tuple(idx)] = 10/neighbor_count + 0.1*dist

        # pick goal with minimum cost
        if len(heuristics) != 0:
            goal_idx = min(heuristics, key=heuristics.get)
        else:
            goal_idx = (0, 0)
        return goal_idx

    def publish_markers(self):
        # markers for frontier points
        frontier_points = self.get_frontier_idxs()
        frontier_points = self.unscale_coordinates(frontier_points)
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.lifetime = rclpy.duration.Duration(seconds=4).to_msg()
        marker.id = 0
        marker.ns = "frontier_markers"

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.r = 0.0
        marker.color.g = 0.4
        marker.color.b = 0.8
        marker.color.a = 1.0

        for p in frontier_points:
            point = Point()
            point.x, point.y = float(p[1]), float(p[0])
            marker.points.append(point)

        self.marker_pub.publish(marker)

        # marker for goal
        goal = self.get_goal(self.odom.pose.pose.position, frontier_points)
        marker1 = Marker()
        marker1.header.frame_id = "map"
        marker1.type = Marker.POINTS
        marker1.action = Marker.ADD
        marker1.header.stamp = self.get_clock().now().to_msg()
        marker1.lifetime = rclpy.duration.Duration(seconds=4).to_msg()
        marker1.id = 1
        marker1.ns = "goal_marker"

        marker1.scale.x = 0.3
        marker1.scale.y = 0.3
        marker1.color.r = 0.0
        marker1.color.g = 1.0
        marker1.color.b = 0.0
        marker1.color.a = 1.0

        goal_point = Point()
        goal_point.x, goal_point.y = float(goal[1]), float(goal[0])
        marker1.points.append(goal_point)

        self.marker_pub.publish(marker1)

    # Shutdown
    def shutdown(self):
        # Destroy the rate and node.
        self.destroy_rate(self.rate)
        self.destroy_node()


    # Run the terminal input loop, send the commands, and ROS spin.
    def loop(self, screen):

        # Initialize the velocity and remaining active time.
        vel = (0.0, 0.0)

        # Run the loop until shutdown.
        forward_vel_multiplier = 1
        std = np.pi/2
        while rclpy.ok():
            # Reduce the active time and stop if necessary.
            vel = (0.0, 0.0)

            # Autonomous decision
            while True: # until finds valid decision
                pos = self.odom.pose.pose.position
                heading = self.odom.pose.pose.orientation.z
                angular_vel_multiplier = random.gauss(0, std)
                angular_vel = angular_vel_multiplier * WNOM
                forward_vel = forward_vel_multiplier * VNOM
                if not self.is_colliding(pos, heading, forward_vel_multiplier,
                                         angular_vel_multiplier, screen):
                    vel = (forward_vel, angular_vel)
                    std = np.pi/2
                    if forward_vel_multiplier < 1:
                        forward_vel_multiplier += 0.005
                    break
                else:
                    print(forward_vel_multiplier)
                    if std < 10*np.pi:
                        std += 0.1
                    if forward_vel_multiplier > -1:
                        forward_vel_multiplier -= 0.1
                    vel = (0.0, 0.0)
                rclpy.spin_once(self)


            # Update the message and publish.
            self.msg.linear.x  = vel[0]
            self.msg.angular.z = vel[1]
            # self.pub.publish(self.msg)

            # print(len(self.get_frontier_idxs()))

            screen.clrtoeol()
            screen.addstr(11, 0,
                        "Sending fwd = %6.3fm/s, spin = %6.3frad/sec" % vel)
            screen.refresh()

            # Spin once to process other items.
            rclpy.spin_once(self)

            # Wait for the next turn.
            # self.rate.sleep()



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
