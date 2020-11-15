"""robot_arm_test controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, DistanceSensor

LIDAR_SENSOR_MAX_RANGE = 3.  # Meters
LIDAR_ANGLE_BINS = 21  # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 1.5708  # 90 degrees, 1.5708 radians

pose_x = 0
pose_y = 0
pose_theta = 0

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())

# get and enable lidar 
lidar = robot.getLidar("LDS-01")
lidar.enable(SIM_TIMESTEP)
lidar.enablePointCloud()

# Initialize lidar motors
lidar_main_motor = robot.getMotor('LDS-01_main_motor')
lidar_secondary_motor = robot.getMotor('LDS-01_secondary_motor')
lidar_main_motor.setPosition(float('inf'))
lidar_secondary_motor.setPosition(float('inf'))
lidar_main_motor.setVelocity(30.0)
lidar_secondary_motor.setVelocity(60.0)

lidar_readings_array = []
start_angle_in_radians = LIDAR_ANGLE_RANGE/2
angle_offset_in_radians = -LIDAR_ANGLE_RANGE/(LIDAR_ANGLE_BINS-1)
lidar_offsets_array = [start_angle_in_radians + angle_offset_in_radians * i for i in range(LIDAR_ANGLE_BINS)]

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(SIM_TIMESTEP) != -1:
    
    lidar_readings_array = lidar.getRangeImage()
    
    print(lidar_readings_array)

# Enter here exit cleanup code.
