from src.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer # Custom replay buffers
from torch.utils.tensorboard import SummaryWriter # For logging to TensorBoard
from src.models.model import Baseline, DoubleDQN # Custom model definitions
import pickle # For object serialization
import torch.nn as nn

class AverageMeter:
    # Computes and stores the average and current value
    def __init__(self) -> None:
        # Initialize and reset the meter
        self.reset()

    def reset(self) -> None:
        # Reset all attributes to initial state
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        # Update the meter with a new value
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MetricLogger:
    def __init__(self, save_dir):
        # Initialize the summary writer and various meters
        self.writer = SummaryWriter(comment=save_dir)
        self.avg_loss = AverageMeter()
        self.avg_Q_max = AverageMeter()
        self.avg_reward = AverageMeter()
        self.avg_score = AverageMeter()
        self.avg_fps = AverageMeter()

    def log_step(self, reward, loss, q):
        # Placeholder for logging step metrics
        pass

    def log_episode(self):
        # Placeholder for logging episode metrics
        pass

    def init_episode(self):
        # Placeholder for initializing an episode
        pass

    def record(self, episode, epsilon, step):
        # Placeholder for recording metrics
        pass

def init_cache(INITIAL_EPSILON, REPLAY_MEMORY, prioritized_replay):
    # Initial variable caching, done only once
    if prioritized_replay:
        # Initialize prioritized replay buffer
        t, D = 0, PrioritizedReplayBuffer(maxlen=REPLAY_MEMORY)
    else:
        # Initialize standard replay buffer
        t, D = 0, ReplayBuffer(maxlen=REPLAY_MEMORY)
    
     # Set up dictionary to store initial settings
    set_up_dict = {"epsilon": INITIAL_EPSILON, "step": t, "D": D, "highest_score": 0}
    save_obj(set_up_dict, "set_up") # Save initial settings

def save_obj(obj, name):
    # Save an object to a file using pickle
    with open('./result/'+ name + '.pkl', 'wb') as f: 
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    # Load an object from a file using pickle
    with open('result/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)