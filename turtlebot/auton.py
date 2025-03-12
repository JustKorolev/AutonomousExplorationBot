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

# ROS Imports
import rclpy

from rclpy.node                 import Node
from rclpy.time                 import Time

from rclpy.qos                  import QoSProfile, DurabilityPolicy
from geometry_msgs.msg          import Twist
from nav_msgs.msg               import Odometry
from nav_msgs.msg               import OccupancyGrid

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

OBSTACLE_THRES = 80

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
        self.odom = Odometry()

        # Create a fixed rate to control the speed of sending commands.
        rate    = 10.0
        self.dt = 1/rate
        self.rate = self.create_rate(rate)

        # Report.
        self.get_logger().info("Auton sending every %f sec..." % self.dt)
        self.get_logger().info("Nominal fwd = %6.3fm/s, spin = %6.3frad/sec"
                               % (self.vnom, self.wnom))

    def updateMap(self, msg):
        self.map = msg

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

        try:
            map_array = np.array(self.map.data, dtype=np.int8).reshape((MAP_HEIGHT, MAP_WIDTH))
        except ValueError:
            return False

        # screen.addstr(16, 0, f"Map stuff: {np.where(map_array > OBSTACLE_THRES)[0]}")
        if np.any(map_array[min_y:max_y+1, min_x:max_x+1] > OBSTACLE_THRES):
            return True  # Collision detected
        return False  # No collision


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
            self.pub.publish(self.msg)

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
