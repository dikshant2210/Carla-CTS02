"""
Author: Dikshant Gupta
Time: 25.07.21 09:57
"""


class Config:
    PI = 3.14159

    simulation_step = 0.05  # 0.008
    sensor_simulation_step = '0.5'
    synchronous = True
    segcam_fov = '90'
    segcam_image_x = '400'  # '1280'
    segcam_image_y = '400'  # '720'

    grid_size = 2  # grid size in meters
    speed_limit = 50
    max_steering_angle = 1.22173  # 70 degrees in radians
    occupancy_grid_width = '1920'
    occupancy_grid_height = '1080'

    location_threshold = 1.0

    ped_speed_range = [0.6, 3.3]
    ped_distance_range = [10, 40]
    car_speed_range = [6, 9]
    scenarios = ['01', '02', '03', '04', '05', '06', '07', '08', '09']  # + ['06', '09']

    val_scenarios = ['01']
    val_ped_speed_range = [0., 4.0]
    val_ped_distance_range = [0, 40]
    val_car_speed_range = [6, 9]

    test_scenarios = ['06']  # ['01', '02', '03', '04']  # ['05', '06', '07', '08', '09']
    test_ped_speed_range = [0.25, 2.85]
    test_ped_distance_range = [4.75, 49.75]
    test_car_speed_range = [6, 9]

    save_freq = 100

    # Setting the SAC training parameters
    batch_size = 2  # 32  # How many experience traces to use for each training step.
    update_freq = 4  # How often to perform a training step after each episode.
    pre_train_steps = 5000  # 10000  # How many steps of random actions before training begins.
    load_model = True  # Whether to load a saved model.
    path = "_out/sac/"  # The path to save our model to.
    total_training_steps = 1000001
    automatic_entropy_tuning = False
    target_update_interval = 1
    hidden_size = 256
    max_epLength = 500  # The max allowed length of our episode.
    sac_gamma = 0.99
    sac_tau = 0.005
    sac_lr = 0.00001
    sac_alpha = 0.1
    num_pedestrians = 4
    num_angles = 5
    num_actions = 3  # num_angles * 3  # acceleration_type
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 500
    episode_buffer = 80
    adrqn_entropy_coef = 0.005
    grad_norm = 0.1

    # angle + 4 car related statistics + 2*num_pedestrians related statistics + one-hot encoded last_action
    input_size = 1 + 4 + 2 * num_pedestrians + num_actions
    image_input_size = 100 * 100 * 3
    tau = 1
    targetUpdateInterval = 10000

    use_dueling = False

    # Simulator Parameters
    host = '127.0.0.1'
    port = 2000
    width = 1280
    height = 720
    display = False
    filter = 'vehicle.audi.tt'
    rolename = 'hero'
    gama = 1.7
    despot_port = 1245
    N_DISCRETE_ACTIONS = 3
    max_speed = 50 * 0.27778  # in m/s
    hit_penalty = 1000
    goal_reward = 1000
    braking_penalty = 1

    # A2C training parameters
    a2c_lr = 0.0001
    a2c_gamma = 0.99
    a2c_gae_lambda = 1.0
    a2c_entropy_coef = 0.005
    a2c_value_loss_coef = 0.5
    max_grad_norm = 50
    num_steps = 500
    train_episodes = 50000
