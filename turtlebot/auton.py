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
from geometry_msgs.msg          import Pose
from nav_msgs.msg               import OccupancyGrid

#
#   Global Definitions
#
VNOM = 0.25
WNOM = 0.5             # This gives a turning radius of v/w = 0.5m

# DIMENSTION IN MAP RESOLUTION
RESOLUTION = 0.05
BOT_WIDTH = 0.24
BOT_LENGTH = 0.40
MAP_WIDTH  = 360
MAP_HEIGHT = 240
ORIGIN_X   = -9.00              # Origin = location of lower-left corner
ORIGIN_Y   = -6.00

OBSTACLE_THRES = 0

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
        self.create_subscription(Pose, '/odom', self.updateOdom, 1)
        self.create_subscription(OccupancyGrid, '/map', self.updateMap, quality)

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
        self.odom = Pose()

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

    def is_colliding(self, pos, step, screen):
        pos_x, pos_y = pos.x, pos.y
        scaled_step = step / RESOLUTION
        new_pos_x, new_pos_y = (pos_x - ORIGIN_X) + scaled_step / RESOLUTION, (pos_y - ORIGIN_Y) / RESOLUTION + scaled_step
        screen.addstr(12, 0,
                        f"Scaled pos: {pos_x, pos_y}")
        print("ebar")
        screen.refresh()
        min_x = max(int(new_pos_x - BOT_WIDTH / 2), 0)
        max_x = min(int(new_pos_x + BOT_WIDTH / 2), MAP_WIDTH - 1)
        min_y = max(int(new_pos_y - BOT_LENGTH / 2), 0)
        max_y = min(int(new_pos_y + BOT_LENGTH / 2), MAP_HEIGHT - 1)

        try:
            map_array = np.array(self.map.data).reshape((MAP_HEIGHT, MAP_WIDTH))
        except ValueError:
            return False

        print(map_array[min_x:max_x+1][min_y:max_y+1])
        if map_array.shape == (MAP_WIDTH, MAP_HEIGHT) and np.any(map_array[min_x:max_x+1][min_y:max_y+1] > OBSTACLE_THRES):
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
        while rclpy.ok():
            # Reduce the active time and stop if necessary.
            step = VNOM * self.dt
            std = np.pi/2
            vel = (0.0, 0.0)

            # Autonomous decision
            while True: # until finds valid decision
                pos = self.odom.position
                print(pos)
                screen.refresh()
                heading_curr = self.odom.orientation.z
                heading_new = random.gauss(heading_curr, std)
                if not self.is_colliding(pos, step, screen):
                    vel = (VNOM, 0.0) # FIXME
                    std = np.pi/2
                    break
                else:
                    std += np.pi/8
                    vel = (0.0, 0.0)


            # Update the message and publish.
            self.msg.linear.x  = vel[0]
            self.msg.angular.z = vel[1]
            self.pub.publish(self.msg)

            # screen.clrtoeol()
            # screen.addstr(10, 0, f"{self.is_colliding(pos, step, screen)}")
            # screen.addstr(11, 0,
            #             "Sending fwd = %6.3fm/s, spin = %6.3frad/sec" % vel)
            # screen.refresh()

            # Spin once to process other items.
            rclpy.spin_once(self)

            # Wait for the next turn.
            self.rate.sleep()



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
