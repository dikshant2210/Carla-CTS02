"""
Author: Dikshant Gupta
Time: 25.07.21 09:57
"""


class Config:
    PI = 3.14159

    simulation_step = 0.1  # 0.008
    sensor_simulation_step = '0.5'
    synchronous = True

    grid_size = 2  # grid size in meters
    speed_limit = 50
    max_steering_angle = 1.22173  # 70 degrees in radians
    occupancy_grid_width = '1920'
    occupancy_grid_height = '1080'

    location_threshold = 1.0

    ped_speed_range = [0.6, 3.3]
    ped_distance_range = [0, 40]
    car_speed_range = [6, 9]
    scenarios = ['01', '03', '04', '07', '08']

    # Setting the SAC training parameters
    batch_size = 32  # How many experience traces to use for each training step.
    trace_length = 8  # How long each experience trace will be when training
    update_freq = 1  # How often to perform a training step.
    y = .995  # Discount factor on the target Q-values
    startE = 1  # Starting chance of random action
    endE = 0.1  # Final chance of random action
    anneling_steps = 200000  # 10000 # How many steps of training to reduce startE to endE.
    num_episodes = 100000  # How many episodes of game environment to train network with.
    pre_train_steps = 10000  # 10000  # How many steps of random actions before training begins.
    load_model = True  # Whether to load a saved model.
    path = "_out/CheckpointsSAC"  # The path to save our model to.
    hidden_size = 256
    var_end_size = 128
    max_epLength = 300  # The max allowed length of our episode.
    time_per_step = 1  # Length of each step used in gif creation
    summaryLength = 100  # Number of episodes to periodically save for analysis
    num_pedestrians = 4
    num_angles = 3
    num_actions = 3  # num_angles * 3  # acceleration_type

    # angle + 4 car related statistics + 2*num_pedestrians related statistics + one-hot encoded last_action
    input_size = 1 + 4 + 2 * num_pedestrians + num_actions
    image_input_size = 100 * 100 * 3
    tau = 1
    targetUpdateInterval = 10000

    use_dueling = False
