args = {
    'algo': 'Baseline',  # algorithm to use: 'Baseline' or 'DoubleDQN'
    'img_channels': 4,  # number of image channels
    'ACTIONS': 3,  # number of possible actions
    'train': 'train',  # mode: 'train' or 'test'
    'game_url': 'http://localhost:8000/t-rex-runner-gh-pages',  # URL of the local game
    'chrome_driver_path': 'path/to/chromedriver',  # path to ChromeDriver
    'init_script': 'path/to/init_script.js',  # path to initialization script for the game
    'EPISODE': 20,  # number of episodes to run
    'FRAME_PER_ACTION': 1,  # number of frames per action
    'lr': 0.00025,  # learning rate
    'weight_decay': 0,  # weight decay for optimizer
    'batch_size': 32,  # batch size
    'gamma': 0.99,  # discount factor
    'grad_norm_clipping': 10  # gradient norm clipping
}
