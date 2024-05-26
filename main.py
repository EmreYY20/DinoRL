from multiprocessing.sharedctypes import Value  # For shared memory data types
from typing import Deque  # For double-ended queue type hints
from utils.env import Game  # Custom game environment class
from types import SimpleNamespace  # For creating simple objects with dynamic attributes
from torch import optim  
from torch.utils.tensorboard import SummaryWriter  # For logging to TensorBoard
from models.model import Baseline, DoubleDQN  # Custom model definitions
from models.train import trainNetwork  # Custom training function
from models.test import test_agent  # Custom testing function
from utils.utils import init_cache, load_obj  # Utility functions for initialization and loading objects
import datetime  # For handling dates and times
import sys  # For system-specific parameters and functions
import importlib  # For importing modules dynamically
import argparse  # For parsing command-line arguments
import torch  
import torch.nn as nn 
import os  # For operating system interfaces

def get_dino_agent(algo):
    # Return the appropriate agent class based on the algorithm name
    if algo == "Baseline":
        print("Using algorithm Baseline.")
        return Baseline
    elif algo == "DoubleDQN":
        print("Using algorithm DoubleDQN.")
        return DoubleDQN
    else:
        raise ValueError

def parse_args():
    # Parse command-line arguments and load configuration
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-c", "--config", help="config filename")
    parser_args, _ = parser.parse_known_args(sys.argv)
    print("Using config file", parser_args.config)

    args = importlib.import_module(parser_args.config).args
    args["experiment_name"] = parser_args.config
    args =  SimpleNamespace(**args) # Convert the arguments to a SimpleNamespace

    return args

# Main function to be run with the script, run with: python main.py -c config
if __name__ == '__main__':
    args = parse_args()

    # Create a log folder for TensorBoard
    if not os.path.isdir('runs'):
        os.makedirs('runs')

    # Create a folder to save the buffer, epsilon for continuous training
    if not os.path.isdir('result'):
        os.makedirs('result')

    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # Create a log directory with the current timestamp
    writer = SummaryWriter(comment=log_dir) # Initialize the TensorBoard writer
    
    DinoAgent = get_dino_agent(args.algorithm)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    agent = DinoAgent(args.img_channels, args.ACTIONS, args.lr, args.weight_decay, 
                      args.BATCH, args.GAMMA, device, args.grad_norm_clipping)
    print("Device:", device)

    if args.train == 'train': # Train a model from scratch
        init_cache(args.INITIAL_EPSILON, args.REPLAY_MEMORY, args.prioritized_replay) # Initialize cache
    else: # Continue training a model or test the agent
        agent = torch.load(args.checkpoint, map_location=device) # Load the model checkpoint
        print ("Weight load successfully")
    
    set_up = load_obj("set_up")
    epsilon, step, Deque, highest_score = set_up['epsilon'], set_up['step'], set_up['D'], set_up['highest_score']
    OBSERVE = args.OBSERVATION
    if args.train == 'test':
        epsilon = 0 # Set epsilon to 0 for testing
        OBSERVE = float('inf') # Set observe to infinity for testing
    
    if args.train != 'test':
        # Initialize the game and start training
        game = Game(args.game_url, args.chrome_driver_path, args.init_script)
        game.screen_shot()

        train = trainNetwork(agent, game, writer, Deque, args.BATCH, device)
        game.press_up() # Start the game
        train.start(epsilon, step, highest_score, 
                OBSERVE, args.ACTIONS, args.EPSILON_DECAY, args.FINAL_EPSILON, 
                args.GAMMA, args.FRAME_PER_ACTION, args.EPISODE, 
                args.SAVE_EVERY, args.SYNC_EVERY, args.TRAIN_EVERY, args.prioritized_replay, args.TEST_EVERY, args)
        game.end()
        print('-------------------------------------Finish Training-------------------------------------')
    else: 
        # Test the agent
        with torch.no_grad():
            test_agent(agent, args, device)
        print('-------------------------------------Finish Testing-------------------------------------')

    print("Exit")