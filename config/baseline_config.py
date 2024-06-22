args = {
    'algo': 'Baseline',  # Algorithm to use: 'Baseline' or 'DoubleDQN'
    'img_channels': 4,  # Number of image channels
    'ACTIONS': 3,  # Number of possible actions
    'train': 'train',  # Mode: 'train' or 'test'
    'game_url': 'http://localhost:8000/t-rex-runner-gh-pages',  # URL of the local game
    'chrome_driver_path': 'path/to/chromedriver',  # Path to ChromeDriver
    'init_script': 'path/to/init_script.js',  # Path to initialization script for the game
    'EPISODE': 20,  # Number of episodes to run
    'FRAME_PER_ACTION': 1,  # Number of frames per action
    'lr': 0.00025,  # Learning rate
    'weight_decay': 0,  # Weight decay for optimizer
    'batch_size': 32,  # Batch size
    'gamma': 0.99,  # Discount factor
    'grad_norm_clipping': 10  # Gradient norm clipping
}
