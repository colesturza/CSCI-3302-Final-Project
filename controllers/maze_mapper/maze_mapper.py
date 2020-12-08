"""
maze_mapper controller.
"""

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
pose_x = 2.125
pose_y = 0.125
pose_theta = math.pi

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


def find_color():
    """
    Determines if the image of the wall in front of the robot is either red, green, blue, or yellow.
    Returns either None (if it is a white wall) or a string for the respective color.
    """
    image = np.array(camera.getImageArray())

    print(image.shape)

    # print(image)

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mask = cv2.inRange(image, [0, 0, 135], [70, 90, 255])
    #
    # _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
    #                                   cv2.CHAIN_APPROX_NONE)
    #
    # blob = max(contours, key=lambda el: cv2.contourArea(el))
    #
    # print(blob)

    # if len(blob) > 100:
    #     return 'Red'

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
