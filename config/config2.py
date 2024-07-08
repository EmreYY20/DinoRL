import os
abs_path = os.path.dirname(__file__)

# training with prioritized experience replay
args = {

    # env setting
    # environment taken from https://github.com/wayou/t-rex-runner
    "game_url": "http://localhost:8000/t-rex-runner-gh-pages", # if local path doesn't work, use website: https://wayou.github.io/t-rex-runner/
    "chrome_driver_path": "/usr/lib/chromium-browser/chromedriver",
    "train": 'train', # options: 'train', 'test'
    "init_script": "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'", # create id for canvas for faster selection from DOM
    "getbase64Script": "canvasRunner = document.getElementById('runner-canvas'); \
    return canvasRunner.toDataURL().substring(22)", # get image from canvas
    "img_rows": 80,
    "img_cols": 80,
    "img_channels": 4, 
    "checkpoint" : "./weights/double_dqn_config2.pth",
    "SAVE_EVERY": 1000, # save model every 1000 episodes
    "num_test_episode": 20, # number of test episodes
    "TEST_EVERY": 1000, # test the model every 1000 episodes
    "SAVE_GIF": False, # save GIF of the gameplay
    "SLEEP": 0.007, # using sleep() to control the FPS in testing

    ################ hyperparameters ################
    "algorithm": "DoubleDQN", # algorithm to use
    "EPISODE": 2500, # number of episodes for training
    "ACTIONS": 2, # possible actions: jump or do nothing
    "GAMMA": 0.99, # decay rate of past observations
    "OBSERVATION": 500, # timesteps to observe before training
    "FINAL_EPSILON": 1e-1, # final value of epsilon (exploration probability)
    "INITIAL_EPSILON": 1, # starting value of epsilon (initial randomness)
    "EPSILON_DECAY": 0.9999925, # decay rate for epsilon
    "REPLAY_MEMORY": 1000, # number of previous transitions to remember
    "BATCH": 32, # size of minibatch
    "FRAME_PER_ACTION": 1, # number of frames per action
    "lr": 1e-4, # learning rate
    "weight_decay": 1e-4, # weight decay for regularization
    "SYNC_EVERY": 1000, # number of experiences between Q_target & Q_online sync
    "TRAIN_EVERY": 3, # train the model every 3 steps
    "prioritized_replay": True, # FPS is slower than unprioritized
    "grad_norm_clipping": 10 # gradient clipping to prevent explosion

}
