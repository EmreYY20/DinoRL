import os
abs_path = os.path.dirname(__file__)

# Standard Training with Initial Configuration
args = {

    # Env setting
    # environment taken from https://github.com/wayou/t-rex-runner
    "game_url": "http://localhost:8000/t-rex-runner-gh-pages", # if local path doesn't work, use website: https://wayou.github.io/t-rex-runner/
    "chrome_driver_path": "/usr/lib/chromium-browser/chromedriver",
    "train": 'test', # Options: 'train', 'test'
    "init_script": "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'", # Create ID for canvas for faster selection from DOM
    "getbase64Script": "canvasRunner = document.getElementById('runner-canvas'); \
    return canvasRunner.toDataURL().substring(22)", # Get image from canvas
    "img_rows": 80,
    "img_cols": 80,
    "img_channels": 4, 
    "checkpoint" : "./weights/double_dqn1.pth",
    "SAVE_EVERY": 1000, # Save model every 1000 episodes
    "num_test_episode": 20, # Number of test episodes
    "TEST_EVERY": 1000, # Test the model every 1000 episodes
    "SAVE_GIF": False, # Save GIF of the gameplay
    "SLEEP": 0.007, # Using sleep() to control the FPS in testing

    # Hyperparameters
    "algorithm": "DoubleDQN", # Algorithm to use
    "EPISODE": 2500, # Number of episodes for training
    "ACTIONS": 2, # Possible actions: jump or do nothing
    "GAMMA": 0.99, # Decay rate of past observations
    "OBSERVATION": 500, # Timesteps to observe before training
    "FINAL_EPSILON": 1e-1, # Final value of epsilon (exploration probability)
    "INITIAL_EPSILON": 1, # Starting value of epsilon (initial randomness)
    "EPSILON_DECAY": 0.9999925, # Decay rate for epsilon
    "REPLAY_MEMORY": 1000, # Number of previous transitions to remember
    "BATCH": 32, # Size of minibatch
    "FRAME_PER_ACTION": 1, # Number of frames per action
    "lr": 1e-4, # Learning rate
    "weight_decay": 1e-4, # Weight decay for regularization
    "SYNC_EVERY": 1000, # Number of experiences between Q_target & Q_online sync
    "TRAIN_EVERY": 3, # Train the model every 3 steps
    "prioritized_replay": False, # FPS is slower than unprioritized
    "grad_norm_clipping": 10 # Gradient clipping to prevent explosion

}