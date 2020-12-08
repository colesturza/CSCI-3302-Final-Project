"""
maze_mapper controller.
"""
import copy
import pickle
import math
import numpy as np
import cv2
from controller import Robot, Motor, DistanceSensor

"""
Initial setup up and global variables.
"""

robot = Robot()  # initialize the robot

state = 'get_target'
sub_state = 'bearing'

LIDAR_SENSOR_MAX_RANGE = 0.25  # Meters
LIDAR_ANGLE_BINS = 21  # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 3.14159  # 180 degrees, 3.14159 radians

# These are your pose values that you will update by solving the odometry equations
pose_x = None
pose_y = None
pose_theta = None

# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# Initialize Motors
leftMotor = robot.getMotor('left wheel motor')
rightMotor = robot.getMotor('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# get and enable gps
gps = robot.getGPS('gps')
gps.enable(SIM_TIMESTEP)

# get and enable compass
compass = robot.getCompass('compass')
compass.enable(SIM_TIMESTEP)

# get and enable lidar
lidar = robot.getLidar('LDS-01')
lidar.enable(SIM_TIMESTEP)
lidar.enablePointCloud()

# Initialize lidar motors
lidar_main_motor = robot.getMotor('LDS-01_main_motor')
lidar_secondary_motor = robot.getMotor('LDS-01_secondary_motor')
lidar_main_motor.setPosition(float('inf'))
lidar_secondary_motor.setPosition(float('inf'))
lidar_main_motor.setVelocity(30.0)
lidar_secondary_motor.setVelocity(60.0)

# Initialize lidar readings array and offsets
lidar_readings_array = []
start_angle_in_radians = LIDAR_ANGLE_RANGE / 2
angle_offset_in_radians = -LIDAR_ANGLE_RANGE / (LIDAR_ANGLE_BINS - 1)
lidar_offsets_array = [start_angle_in_radians + angle_offset_in_radians * i for i in range(LIDAR_ANGLE_BINS)]

CENTER_LIDAR_IDX = 10
LEFT_LIDAR_IDX = 0
RIGHT_LIDAR_IDX = 20

# Map Variables
MAP_BOUNDS = [1.5, 1.5]
CELL_RESOLUTIONS = np.array([0.25, 0.25])  # 25cm per cell
NUM_X_CELLS = int(MAP_BOUNDS[0] / CELL_RESOLUTIONS[0])
NUM_Y_CELLS = int(MAP_BOUNDS[1] / CELL_RESOLUTIONS[1])

world_map = np.zeros([NUM_Y_CELLS, NUM_X_CELLS, 6])

# Set the boundary walls
world_map[:, NUM_X_CELLS-1, 1] = 1  # North facing wall
world_map[0, :, 2] = 1  # West facing wall
world_map[:, 0, 3] = 1  # South facing wall
world_map[NUM_Y_CELLS-1, :, 4] = 1  # East facing wall

# Set starting spot to visited
world_map[5, 0, 0] = 1

camera = robot.getCamera('camera')
camera.enable(SIM_TIMESTEP)

"""
End of setup and global variables.
"""

color_ranges = []


def add_color_range_to_detect(lower_bound, upper_bound):
    """
    @param lower_bound: Tuple of BGR values
    @param upper_bound: Tuple of BGR values
    """
    global color_ranges
    color_ranges.append([lower_bound, upper_bound])  # Add color range to global list of color ranges to detect


def check_if_color_in_range(bgr_tuple):
    """
    @param bgr_tuple: Tuple of BGR values
    @returns Boolean: True if bgr_tuple is in any of the color ranges specified in color_ranges
    """
    global color_ranges
    for entry in color_ranges:
        lower, upper = entry[0], entry[1]
        in_range = True
        for i in range(len(bgr_tuple)):
            if bgr_tuple[i] < lower[i] or bgr_tuple[i] > upper[i]:
                in_range = False
                break
        if in_range:
            return True

    return False


def do_color_filtering(img):
    # Color Filtering
    # Objective: Take an RGB image as input, and create a "mask image" to filter out irrelevant pixels
    # Definition "mask image":
    #    An 'image' (really, a matrix) of 0s and 1s, where 0 indicates that the corresponding pixel in
    #    the RGB image isn't important (i.e., what we consider background) and 1 indicates foreground.
    #
    #    Importantly, we can multiply pixels in an image by those in a mask to 'cancel out' all of the pixels we don't
    #    care about. Once we've done that step, the only non-zero pixels in our image will be foreground
    #
    # Approach:
    # Create a mask image: a matrix of zeros ( using np.zeroes([height width]) ) of the same height and width as
    # the input image.
    # For each pixel in the input image, check if it's within the range of allowable colors for your detector
    #     If it is: set the corresponding entry in your mask to 1
    #     Otherwise: set the corresponding entry in your mask to 0 (or do nothing, since it's initialized to 0)
    # Return the mask image
    img_height = img.shape[0]
    img_width = img.shape[1]

    # Create a matrix of dimensions [height, width] using numpy
    mask = np.zeros([img_height, img_width])  # Index mask as [height, width] (e.g.,: mask[y,x])

    for i in range(img_height):

        for j in range(img_width):

            if check_if_color_in_range(img[i, j]):

                mask[i, j] = 1

    return mask


def expand_nr(img_mask, cur_coord, coordinates_in_blob):
    # Non-recursive function to find all of the non-zero pixels connected to a location

    # If value of img_mask at cur_coordinate is 0, or cur_coordinate is out of bounds (either x,y < 0 or x,y >= width
    # or height of img_mask) return and stop expanding
    # Otherwise, add this to our blob:
    # Set img_mask at cur_coordinate to 0 so we don't double-count this coordinate if we expand
    # back onto it in the future
    # Add cur_coordinate to coordinates_in_blob
    # Call expand on all 4 neighboring coordinates of cur_coordinate (above/below, right/left). Make sure
    # you pass in the same img_mask and coordinates_in_blob objects you were passed so the recursive calls all share
    # the same objects

    coordinate_list = [cur_coord]  # List of all coordinates to try expanding to
    while len(coordinate_list) > 0:
        cur_coordinate = coordinate_list.pop()  # Take the first coordinate in the list and perform 'expand' on it
        if cur_coordinate[0] < 0 or cur_coordinate[1] < 0 or cur_coordinate[0] >= img_mask.shape[0] \
                or cur_coordinate[1] >= img_mask.shape[1]:
            continue
        if img_mask[cur_coordinate[0], cur_coordinate[1]] == 0.0:
            continue

        img_mask[cur_coordinate[0], cur_coordinate[1]] = 0
        coordinates_in_blob.append(cur_coordinate)

        coordinate_list.append((cur_coordinate[0] + 1, cur_coordinate[1]))
        coordinate_list.append((cur_coordinate[0] - 1, cur_coordinate[1]))
        coordinate_list.append((cur_coordinate[0], cur_coordinate[1] - 1))
        coordinate_list.append((cur_coordinate[0], cur_coordinate[1] + 1))


def get_blobs(img_mask):
    # Blob detection
    # Objective: Take a mask image as input, group each blob of non-zero pixels as a detected object,
    #            and return a list of lists containing the coordinates of each pixel belonging to each blob.
    # Recommended Approach:
    # Create a copy of the mask image so you can edit it during blob detection
    # Create an empty blobs_list to hold the coordinates of each blob's pixels
    # Iterate through each coordinate in the mask:
    #   If you find an entry that has a non-zero value:
    #     Create an empty list to store the pixel coordinates of the blob
    #     Call the recursive "expand" function on that position, recording coordinates of non-zero
    #     pixels connected to it
    #     Add the list of coordinates to your blobs_list variable
    # Return blobs_list

    img_mask_height = img_mask.shape[0]
    img_mask_width = img_mask.shape[1]

    img_mask_copy = copy.copy(img_mask)

    blobs_list = []  # List of all blobs, each element being a list of coordinates belonging to each blob

    for i in range(img_mask_height):

        for j in range(img_mask_width):

            if img_mask[i, j] == 1.0:
                blob_coords = []
                expand_nr(img_mask_copy, (i, j), blob_coords)
                blobs_list.append(blob_coords)

    return blobs_list


def find_color():
    """
    Determines if the image of the wall in front of the robot is either red, green, blue, or yellow.
    Returns either None (if it is a white wall) or a string for the respective color.
    """
    global pose_x, pose_y
    image = np.array(camera.getImageArray(), dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mask = do_color_filtering(image)
    blobs = get_blobs(mask)

    if blobs:
        blob = max(blobs, key=lambda el: len(el))

        if len(blob) > 100:
            return True

    # # Check for red
    # red_mask = cv2.inRange(image, np.array([0., 0., 40.]), np.array([70., 90., 255.]))
    # contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    # if transform_world_coord_to_map_coord((pose_x, pose_y)) == (4, 3):
    #     print(red_mask)
    #     print("At red tile")
    #     pickle.dump(image, open('red_image', 'wb'))
    #     cv2.imshow('red', image)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    #
    # if contours:
    #     blob = max(contours, key=lambda el: cv2.contourArea(el))
    #     if len(blob) > 250:
    #         print('Red')
    #         return 'Red'
    #
    # # Check for green
    # green_mask = cv2.inRange(image, np.array([15, 110, 0]), np.array([130, 240, 110]))
    # contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    # if transform_world_coord_to_map_coord((pose_x, pose_y)) == (2, 4):
    #     print("At green tile")
    #     pickle.dump(image, open('green_image', 'wb'))
    #     cv2.imshow('green', image)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    #
    # if contours:
    #     blob = max(contours, key=lambda el: cv2.contourArea(el))
    #     if len(blob) > 250:
    #         print('Green')
    #         return 'Green'
    #
    # # Check for blue
    # blue_mask = cv2.inRange(image, np.array([40, 0, 0]), np.array([255, 140, 140]))
    # contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    # if transform_world_coord_to_map_coord((pose_x, pose_y)) == (1, 1):
    #     print("At blue tile")
    #     pickle.dump(image, open('blue_image', 'wb'))
    #     cv2.imshow('blue', image)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    #
    # if contours:
    #     blob = max(contours, key=lambda el: cv2.contourArea(el))
    #     if len(blob) > 250:
    #         print('Blue')
    #         return 'Blue'

    return None


def get_bounded_theta(theta):
    """
    Returns theta bounded in [-PI, PI]
    """
    while theta > math.pi:
        theta -= 2. * math.pi
    while theta < -math.pi:
        theta += 2. * math.pi
    return theta


def transform_world_coord_to_map_coord(world_coord):
    """
    @param world_coord: Tuple of (x,y) position in world coordinates
    @return grid_coord: Tuple of (i,j) coordinates corresponding to grid row (y-coord) and column (x-coord) in our map
    """
    col, row = np.array(world_coord) / CELL_RESOLUTIONS
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None

    return tuple(np.array([row, col]).astype(int))


def transform_map_coord_world_coord(map_coord):
    """
    @param map_coord: Tuple of (i,j) coordinates corresponding to grid column and row in our map
    @return world_coord: Tuple of (x,y) position corresponding to the center of map_coord, in world coordinates
    """
    row, col = map_coord
    if row < 0 or col < 0 or row >= NUM_Y_CELLS or col >= NUM_X_CELLS:
        return None

    return np.array([(col + 0.5) * CELL_RESOLUTIONS[1], (row + 0.5) * CELL_RESOLUTIONS[0]])


def update_map():
    """
    Updates the world map using the lidar sensor readings and determines if the wall in front of the
    robot is a color of interest.
    """
    global pose_x, pose_y, pose_theta
    global lidar_readings_array, CENTER_LIDAR_IDX, LEFT_LIDAR_IDX, RIGHT_LIDAR_IDX
    global world_map

    map_coords = transform_world_coord_to_map_coord((pose_x, pose_y))

    # set the current tile to visited
    if world_map[map_coords][0] == 0:
        world_map[map_coords][0] = 1

    facing = None

    facing_degree = pose_theta * 180 / math.pi  # convert to degrees for readability

    # Determine the directional facing
    if math.isclose(facing_degree, 90, abs_tol=15):
        facing = 'North'

    elif math.isclose(facing_degree, 360, abs_tol=15) or math.isclose(facing_degree, 0, abs_tol=15):
        facing = 'East'

    elif math.isclose(facing_degree, 270, abs_tol=15):
        facing = 'South'

    elif math.isclose(facing_degree, 180, abs_tol=15):
        facing = 'West'

    # check for the presence of a wall
    center = True if lidar_readings_array[CENTER_LIDAR_IDX] <= LIDAR_SENSOR_MAX_RANGE else False
    left = True if lidar_readings_array[LEFT_LIDAR_IDX] <= LIDAR_SENSOR_MAX_RANGE else False
    right = True if lidar_readings_array[RIGHT_LIDAR_IDX] <= LIDAR_SENSOR_MAX_RANGE else False

    if center:
        color = find_color()  # check if the wall is a color of interest

        if color is not None:
            print(color)

        if facing == 'North':
            world_map[map_coords][1] = 1
        elif facing == 'West':
            world_map[map_coords][2] = 1
        elif facing == 'South':
            world_map[map_coords][3] = 1
        elif facing == 'East':
            world_map[map_coords][4] = 1

    if left:
        if facing == 'North':
            world_map[map_coords][2] = 1  # Left sensor is facing West
        elif facing == 'West':
            world_map[map_coords][3] = 1  # Left sensor is facing South
        elif facing == 'South':
            world_map[map_coords][4] = 1  # Left sensor is facing East
        elif facing == 'East':
            world_map[map_coords][1] = 1  # Left sensor is facing North

    if right:
        if facing == 'North':
            world_map[map_coords][4] = 1  # Right sensor is facing East
        elif facing == 'West':
            world_map[map_coords][1] = 1  # Right sensor is facing North
        elif facing == 'South':
            world_map[map_coords][2] = 1  # Right sensor is facing South
        elif facing == 'East':
            world_map[map_coords][3] = 1  # Right sensor is facing West


def get_next_target():
    """
    Randomly pick a new target from the current location, heavily weight unvisited targets.
    Returns None if all targets have been visited.
    """
    global pose_x, pose_y, pose_theta
    global world_map

    # we visited all the targets already
    if world_map[:, :, 0].all():
        return None

    map_coords = transform_world_coord_to_map_coord((pose_x, pose_y))

    x, y = map_coords

    candidate_targets = []
    weights = []

    # check north tile
    if world_map[x, y][1] != 1:
        candidate_targets.append((x, y + 1))

        if world_map[x, y + 1][0] == 0:
            weights.append(100)
        else:
            weights.append(1)

    # check west tile
    if world_map[x, y][2] != 1:
        candidate_targets.append((x - 1, y))

        if world_map[x - 1, y][0] == 0:
            weights.append(100)
        else:
            weights.append(1)

    # check south tile
    if world_map[x, y][3] != 1:
        candidate_targets.append((x, y - 1))

        if world_map[x, y - 1][0] == 0:
            weights.append(100)
        else:
            weights.append(1)

    # check east tile
    if world_map[x, y][4] != 1:
        candidate_targets.append((x + 1, y))

        if world_map[x + 1, y][0] == 0:
            weights.append(100)
        else:
            weights.append(1)

    weights = np.array(weights) / np.sum(weights)  # convert the weights to percentages

    new_target_idx = np.random.choice(range(len(candidate_targets)), p=weights)  # choose at random based on weights

    print(np.sum(world_map[:, :, 0]))  # output the total number of tiles visited

    return candidate_targets[new_target_idx]


def world_map_to_graph():
    """
    Convert the world map to a graph, represented as an adjacency list.
    """
    global world_map
    graph_dict = {}

    for i in range(len(world_map)):
        for j in range(len(world_map[0])):

            graph_dict[(i, j)] = []

            # if not visited, do nothing
            if world_map[i, j, 0] == 0:
                continue

            # if no north wall, add adjacency to north tile
            if world_map[i, j, 1] == 0:
                graph_dict[(i, j)].append((i, j + 1))

            # if no west wall, add adjacency to west tile
            if world_map[i, j, 2] == 0:
                graph_dict[(i, j)].append((i - 1, j))

            # if no south wall, add adjacency to south tile
            if world_map[i, j, 3] == 0:
                graph_dict[(i, j)].append((i, j - 1))

            # if no east wall, add adjacency to east tile
            if world_map[i, j, 4] == 0:
                graph_dict[(i, j)].append((i + 1, j))

    return graph_dict


def find_shortest_path(graph_dict, start, goal):
    """
    Find the shortest path between two nodes using BFS.
    """
    dist = {}
    prev = {}

    for key in graph_dict.keys():
        dist[key] = math.inf
        prev[key] = None

    dist[start] = 0

    q_cost = dist.copy()

    while len(q_cost) > 0:
        u = min(q_cost, key=q_cost.get)
        _ = q_cost.pop(u)
        for neighbor in graph_dict[u]:
            new_dist = dist[u] + 1
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                q_cost[neighbor] = new_dist
                prev[neighbor] = u

    path = [goal]
    curr = prev[goal]
    while curr != start:
        path.append(curr)
        curr = prev[curr]
    path.append(curr)

    return path[::-1]


"""
Start the simulation.
"""


def main():
    global robot, state, sub_state, world_map
    global leftMotor, rightMotor, SIM_TIMESTEP
    global pose_x, pose_y, pose_theta
    global lidar_readings_array

    add_color_range_to_detect([0, 0, 135], [70, 90, 255])  # Detect red
    add_color_range_to_detect([15, 110, 0], [130, 240, 110])  # Detect green
    add_color_range_to_detect([130, 0, 0], [255, 140, 140])  # Detect blue
    add_color_range_to_detect([0, 130, 200], [60, 255, 255])  # Detect yellow

    for i in range(10):
        robot.step(SIM_TIMESTEP)

    first_loop = True

    target_bearing = None
    target_pose = None

    # Main Control Loop:
    while robot.step(SIM_TIMESTEP) != -1:

        # update odometry
        pose_x, _, pose_y = gps.getValues()
        compass_values = compass.getValues()
        pose_theta = math.atan2(compass_values[2], compass_values[0])  # get_bearing_in_degrees(compass_values)

        if pose_theta >= 0:
            pose_theta = pose_theta
        else:
            pose_theta = (2 * math.pi + pose_theta)

        lidar_readings_array = lidar.getRangeImage()  # get lidar readings

        # Update the walls at the starting location
        if first_loop:
            update_map()
            first_loop = False

        # get the next target in the mapping phase
        if state == "get_target":

            # get the next target and determine if the entire maze has been searched
            target_pose_map_coords = get_next_target()
            if target_pose_map_coords is None:
                # mapping is complete
                leftMotor.setVelocity(0)
                rightMotor.setVelocity(0)
                print('Here')
                break

            target_pose = transform_map_coord_world_coord(target_pose_map_coords)  # retrieve the target world coords
            # not entirely sure why we had to switch the x and y, but it works this way.
            target_bearing = math.atan2(target_pose[0] - pose_x, target_pose[1] - pose_y)  # find bearing error

            if target_bearing >= 0:
                target_bearing = target_bearing
            else:
                target_bearing = (2 * math.pi + target_bearing)

            state = "turn_drive_turn_control"

        # move to the target location
        elif state == "turn_drive_turn_control":

            bearing_error = pose_theta - target_bearing
            distance_error = np.linalg.norm(np.array(target_pose) - np.array([pose_x, pose_y]))

            # decrease the bearing error
            if sub_state == "bearing":
                if bearing_error > 0.005:
                    leftMotor.setVelocity(leftMotor.getMaxVelocity() * 0.05)
                    rightMotor.setVelocity(-rightMotor.getMaxVelocity() * 0.05)
                elif bearing_error < -0.005:
                    leftMotor.setVelocity(-leftMotor.getMaxVelocity() * 0.05)
                    rightMotor.setVelocity(rightMotor.getMaxVelocity() * 0.05)
                else:
                    leftMotor.setVelocity(0)
                    rightMotor.setVelocity(0)
                    sub_state = "distance"
            # decrease the distance error
            elif sub_state == "distance":
                if distance_error > 0.025:
                    leftMotor.setVelocity(leftMotor.getMaxVelocity() * 0.2)
                    rightMotor.setVelocity(rightMotor.getMaxVelocity() * 0.2)
                else:
                    leftMotor.setVelocity(0)
                    rightMotor.setVelocity(0)

                    update_map()  # update the map once in the center of the target tile

                    sub_state = "bearing"
                    state = "get_target"

    print(world_map[:, :, 0])
    print(world_map[:, :, 1])
    print(world_map[:, :, 2])
    print(world_map[:, :, 3])
    print(world_map[:, :, 4])

    graph = world_map_to_graph()
    path = find_shortest_path(graph, (0, 0), (5, 3))

    print(graph)
    print(path)

    # Enter here exit cleanup code.


if __name__ == "__main__":
    main()
