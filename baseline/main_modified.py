############ import libraries ############

import sys
import os
from types import SimpleNamespace
from src.env import Game
from torch.utils.tensorboard import SummaryWriter
from src.models.model import DoubleDQN
from baseline.train_modified import trainNetwork
from misc.utils import init_cache, load_obj
import importlib
import argparse
import torch

#########################################

# add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_dino_agent(algo):
    if algo == "Baseline":
        print("Using Baseline.")
        return DoubleDQN
    else:
        raise ValueError

def parse_args():
    parser = argparse.ArgumentParser(description='DinoRL Training Script')
    parser.add_argument("-c", "--config", required=True, help="Path to the configuration file")
    parser_args = parser.parse_args()

    if parser_args.config is None:
        raise ValueError("Configuration file path must be provided using the -c or --config argument.")

    print("Using config file", parser_args.config)

    config_dir = os.path.dirname(parser_args.config)
    sys.path.insert(0, config_dir)

    config_module_name = os.path.basename(parser_args.config).replace('.py', '')
    config_module = importlib.import_module(config_module_name)

    args = SimpleNamespace(**config_module.args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    required_attributes = [
        'algo', 'img_channels', 'ACTIONS', 'train', 
        'game_url', 'chrome_driver_path', 'init_script',
        'EPISODE', 'FRAME_PER_ACTION', 'lr', 'weight_decay', 
        'batch_size', 'gamma', 'grad_norm_clipping'
    ]

    for attr in required_attributes:
        if not hasattr(args, attr):
            raise AttributeError(f"Missing required config attribute: {attr}")

    return args

def main():
    args = parse_args()  # parse command-line arguments
    writer = SummaryWriter()  # initialize tensorboard writer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = get_dino_agent(args.algo)(
        args.img_channels, args.ACTIONS, args.lr, args.weight_decay, 
        args.batch_size, args.gamma, args.device, args.grad_norm_clipping
    )
    print("Device:", device)

    print('-------------------------------------Start Training-------------------------------------')
    game = Game(args.game_url, args.chrome_driver_path, args.init_script)  # initialize game environment
    game.screen_shot()  # take initial screenshot
    train = trainNetwork(agent, game, device)  # initialize training
    game.press_up()  # start game
    train.start(args.EPISODE, args.ACTIONS, args.FRAME_PER_ACTION)  # begin training
    game.end()  # end game session
    print('-------------------------------------Finish Training-------------------------------------')

if __name__ == "__main__":
    main()
