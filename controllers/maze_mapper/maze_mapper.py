"""maze_mapper controller."""

import math
import numpy as np
from controller import Robot, Motor, DistanceSensor

import supervisor

# supervisor.init_supervisor()
# robot = supervisor.supervisor

robot = Robot()

LIDAR_SENSOR_MAX_RANGE = 0.25  # Meters
LIDAR_ANGLE_BINS = 21  # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 3.14159  # 180 degrees, 3.14159 radians

# These are your pose values that you will update by solving the odometry equations
pose_x = 2.125
pose_y = 0.125
pose_theta = math.pi

# velocity reduction percent
MAX_VEL_REDUCTION = 1  # Run robot at 20% of max speed

# ePuck Constants
EPUCK_AXLE_DIAMETER = 0.053  # ePuck's wheels are 53mm apart.
EPUCK_MAX_WHEEL_SPEED = 0.125 * MAX_VEL_REDUCTION  # To be filled in with ePuck wheel speed in m/s

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
MAP_BOUNDS = [2.25, 2.25]
CELL_RESOLUTIONS = np.array([0.25, 0.25])  # 10cm per cell
NUM_X_CELLS = int(MAP_BOUNDS[0] / CELL_RESOLUTIONS[0])
NUM_Y_CELLS = int(MAP_BOUNDS[1] / CELL_RESOLUTIONS[1])

world_map = np.zeros([NUM_Y_CELLS, NUM_X_CELLS, 6])

# Set the boundary walls
world_map[:, 8, 1] = 1  # North facing wall
world_map[0, :, 2] = 1  # West facing wall
world_map[:, 0, 3] = 1  # South facing wall
world_map[8, :, 4] = 1  # East facing wall


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

    return np.array([(col+0.5)*CELL_RESOLUTIONS[1], (row+0.5)*CELL_RESOLUTIONS[0]])


def update_map():
    global pose_x, pose_y, pose_theta
    global lidar_readings_array, CENTER_LIDAR_IDX, LEFT_LIDAR_IDX, RIGHT_LIDAR_IDX
    global world_map

    map_coords = transform_world_coord_to_map_coord((pose_x, pose_y))
    if world_map[map_coords][0] == 0:
        world_map[map_coords][0] = 1

    facing = None

    if math.isclose(pose_theta, 90, abs_tol=10):
        facing = 'North'

    elif math.isclose(pose_theta, 360, abs_tol=10):
        facing = 'West'

    elif math.isclose(pose_theta, 270, abs_tol=10):
        facing = 'South'

    elif math.isclose(pose_theta, 180, abs_tol=10):
        facing = 'East'

    center = True if lidar_readings_array[CENTER_LIDAR_IDX] <= LIDAR_SENSOR_MAX_RANGE else False
    left = True if lidar_readings_array[LEFT_LIDAR_IDX] <= LIDAR_SENSOR_MAX_RANGE else False
    right = True if lidar_readings_array[RIGHT_LIDAR_IDX] <= LIDAR_SENSOR_MAX_RANGE else False

    if center:
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


def get_bearing_in_degrees(values):
    rad = math.atan2(compass_values[0], compass_values[2])
    bearing = (rad - 1.5708) / math.pi * 180.0;
    if bearing < 0.0:
        bearing = bearing + 360.0
    return bearing


def get_target_bearing(to_coords, from_coords):
    """
    Uses world map coordinates to get the target bearing in degrees
    """
    direction = (to_coords[0] - from_coords[0], to_coords[1] - from_coords[1])

    print(direction, to_coords, from_coords)

    if direction == (-1, 0):
        return 360
    if direction == (1, 0):
        return 180
    if direction == (0, 1):
        return 90
    return 270


# Important IK Variable storing final desired pose
target_poses = [(7, 0), (6, 0), (5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0)]  # Populated by the supervisor, only when the target is moved.
target_pose = None
target_bearing = None

# Sensor burn-in period
for i in range(10):
    robot.step(SIM_TIMESTEP)

state = 'get_target'
sub_state = 'bearing'

# Main Control Loop:
while robot.step(SIM_TIMESTEP) != -1:

    pose_x, _, pose_y = gps.getValues()
    compass_values = compass.getValues()
    pose_theta = get_bearing_in_degrees(compass_values)

    lidar_readings_array = lidar.getRangeImage()

    if state == "get_target":

        if not target_poses:
            break

        target_pose_map_coords = target_poses.pop(0)
        current_pose_map_coords = transform_world_coord_to_map_coord((pose_x, pose_y))

        target_bearing = get_target_bearing(target_pose_map_coords, current_pose_map_coords)
        target_pose = transform_map_coord_world_coord(target_pose_map_coords)

        print(target_pose, target_bearing)
        state = "turn_drive_turn_control"

    elif state == "turn_drive_turn_control":

        bearing_error = abs(target_bearing - pose_theta)
        distance_error = np.linalg.norm(np.array(target_pose) - np.array([pose_x, pose_y]))

        if sub_state == "bearing":
            if bearing_error > 1:
                leftMotor.setVelocity(-leftMotor.getMaxVelocity() * 0.1)
                rightMotor.setVelocity(rightMotor.getMaxVelocity() * 0.1)
            else:
                leftMotor.setVelocity(0)
                rightMotor.setVelocity(0)
                sub_state = "distance"
        elif sub_state == "distance":
            if distance_error > 0.01:
                leftMotor.setVelocity(leftMotor.getMaxVelocity() * 0.2)
                rightMotor.setVelocity(rightMotor.getMaxVelocity() * 0.2)
            else:
                leftMotor.setVelocity(0)
                rightMotor.setVelocity(0)

                update_map()

                sub_state = "bearing"
                state = "get_target"

print(world_map[:, :, 0])
print(world_map[:, :, 1])
print(world_map[:, :, 2])
print(world_map[:, :, 3])
print(world_map[:, :, 4])

# Enter here exit cleanup code.
