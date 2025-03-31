from shapely.geometry import LineString
from math import dist
import numpy as np

MAP_WIDTH  = 360
MAP_HEIGHT = 240
OBSTACLE_THRESH = 80
CLEAR_THRESH = 30
BOT_WIDTH_CLEARANCE = 10
BOT_LENGTH_CLEARANCE = 10


class Node:
    def __init__(self, x, y):
        # Define a parent (cleared for now).
        self.parent = None

        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    def bresenham(self, start, end):
        # Extract the coordinates
        (xs, ys) = start
        (xe, ye) = end

        # if (xs == xe):
        #     print("HEREREREE")

        # Move along ray (excluding endpoint).
        if (np.abs(xe-xs) >= np.abs(ye-ys)):
            return[(u, int(ys + (ye-ys)/(xe-xs) * (u+0.5-xs)))
                   for u in range(int(xs), int(xe), int(np.sign(xe-xs)))]
        else:
            return[(int(xs + (xe-xs)/(ye-ys) * (v+0.5-ys)), v)
                   for v in range(int(ys), int(ye), np.sign(ye-ys))]

    # Compute/create an intermediate node.  This can be useful if you
    # need to check the local planner by testing intermediate nodes.
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y))

    # Return a tuple of coordinates, used to compute Euclidean distance.
    def coordinates(self):
        return (self.x, self.y)

    # Compute the Euclidean distance to another node.
    def distance(self, other):
        return np.linalg.norm(np.subtract(self.coordinates(), other.coordinates()))

    ################
    # Collision functions:
    # Check whether in free space.
    def inFreespace(self, map_array):
        if (self.x <= 0 or self.x >= MAP_WIDTH or
            self.y <= 0 or self.y >= MAP_HEIGHT):
            return False
        x, y = self.coordinates()
        min_x = max(int(x - BOT_WIDTH_CLEARANCE / 2), 0)
        max_x = min(int(x + BOT_WIDTH_CLEARANCE / 2), MAP_WIDTH - 1)
        min_y = max(int(y - BOT_LENGTH_CLEARANCE / 2), 0)
        max_y = min(int(y + BOT_LENGTH_CLEARANCE / 2), MAP_HEIGHT - 1)
        return np.any(map_array[min_y:max_y+1, min_x:max_x+1] < OBSTACLE_THRESH)

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other, map_array):
        intermediates = np.array(self.bresenham(self.coordinates(),
                                 other.coordinates()))

        for x_interm, y_interm in intermediates:
            min_x = max(int(x_interm - BOT_WIDTH_CLEARANCE / 2), 0)
            max_x = min(int(x_interm + BOT_WIDTH_CLEARANCE / 2), MAP_WIDTH - 1)
            min_y = max(int(y_interm - BOT_LENGTH_CLEARANCE / 2), 0)
            max_y = min(int(y_interm + BOT_LENGTH_CLEARANCE / 2), MAP_HEIGHT - 1)
            if np.any(map_array[min_y:max_y+1, min_x:max_x+1] > OBSTACLE_THRESH):
                return False
        return True