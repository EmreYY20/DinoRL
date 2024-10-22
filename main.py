############ import libraries ############

from multiprocessing.sharedctypes import Value
from typing import Deque
from src.env import Game
from types import SimpleNamespace
from torch import optim  
from torch.utils.tensorboard import SummaryWriter
from src.models.model import DoubleDQN
from src.models.train import trainNetwork
from src.models.test import test_agent
from misc.utils import init_cache, load_obj
import datetime
import sys
import importlib
import argparse
import torch  
import torch.nn as nn 
import os

#########################################

def get_dino_agent(algo):
    # return the appropriate agent class based on the algorithm name
    if algo == "DoubleDQN":
        print("Using algorithm DoubleDQN.")
        return DoubleDQN
    else:
        raise ValueError

def parse_args():
    # parse command-line arguments and load configuration
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-c", "--config", help="config filename")
    parser_args, _ = parser.parse_known_args(sys.argv)
    print("Using config file", parser_args.config)

    # add the config directory to sys.path
    config_dir = os.path.dirname(parser_args.config)
    sys.path.insert(0, config_dir)

    # import the config module
    config_module_name = os.path.basename(parser_args.config).replace('.py', '')
    config_module = importlib.import_module(config_module_name)

    args = config_module.args

    #args = importlib.import_module(parser_args.config).args
    args["experiment_name"] = parser_args.config
    args = SimpleNamespace(**args) # convert the arguments to a SimpleNamespace

    return args

# main function to be run with the script, run with: python main.py -c config
if __name__ == '__main__':
    args = parse_args()

    # create a log folder for TensorBoard
    if not os.path.isdir('runs'):
        os.makedirs('runs')

    # create a folder to save the buffer, epsilon for continuous training
    if not os.path.isdir('result'):
        os.makedirs('result')

    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # create a log directory with the current timestamp
    writer = SummaryWriter(comment=log_dir) # initialize the TensorBoard writer
    
    DinoAgent = get_dino_agent(args.algorithm)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    agent = DinoAgent(args.img_channels, args.ACTIONS, args.lr, args.weight_decay, 
                      args.BATCH, args.GAMMA, device, args.grad_norm_clipping)
    print("Device:", device)

    if args.train == 'train': # train a model from scratch
        init_cache(args.INITIAL_EPSILON, args.REPLAY_MEMORY, args.prioritized_replay) # initialize cache
    else: # continue training a model or test the agent
        agent = torch.load(args.checkpoint, map_location=device) # load the model checkpoint
        print ("Weight load successfully")
    
    set_up = load_obj("set_up")
    epsilon, step, Deque, highest_score = set_up['epsilon'], set_up['step'], set_up['D'], set_up['highest_score']
    OBSERVE = args.OBSERVATION
    if args.train == 'test':
        epsilon = 0 # set epsilon to 0 for testing
        OBSERVE = float('inf') # set observe to infinity for testing
    
    if args.train != 'test':
        # initialize the game and start training
        print('-------------------------------------Start Training-------------------------------------')
        game = Game(args.game_url, args.chrome_driver_path, args.init_script)
        game.screen_shot()

        train = trainNetwork(agent, game, writer, Deque, args.BATCH, device)
        game.press_up() # start the game
        train.start(epsilon, step, highest_score, 
                OBSERVE, args.ACTIONS, args.EPSILON_DECAY, args.FINAL_EPSILON, 
                args.GAMMA, args.FRAME_PER_ACTION, args.EPISODE, 
                args.SAVE_EVERY, args.SYNC_EVERY, args.TRAIN_EVERY, args.prioritized_replay, args.TEST_EVERY, args)
        game.end()
        print('-------------------------------------Finish Training-------------------------------------')
    else: 
        # test the agent
        with torch.no_grad():
            test_agent(agent, args, device)
        print('-------------------------------------Finish Testing-------------------------------------')
