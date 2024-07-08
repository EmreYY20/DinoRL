############ import libraries ############

from src.ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer 
from torch.utils.tensorboard import SummaryWriter
from src.models.model import DoubleDQN 
import pickle 
import torch.nn as nn

#########################################

class AverageMeter:
    # computes and stores the average and current value
    def __init__(self) -> None:
        # initialize and reset the meter
        self.reset()

    def reset(self) -> None:
        # reset all attributes to initial state
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        # update the meter with a new value
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MetricLogger:
    def __init__(self, save_dir):
        # initialize the summary writer and various meters
        self.writer = SummaryWriter(comment=save_dir)
        self.avg_loss = AverageMeter()
        self.avg_Q_max = AverageMeter()
        self.avg_reward = AverageMeter()
        self.avg_score = AverageMeter()
        self.avg_fps = AverageMeter()

    def log_step(self, reward, loss, q):
        # placeholder for logging step metrics
        pass

    def log_episode(self):
        # placeholder for logging episode metrics
        pass

    def init_episode(self):
        # placeholder for initializing an episode
        pass

    def record(self, episode, epsilon, step):
        # placeholder for recording metrics
        pass

def init_cache(INITIAL_EPSILON, REPLAY_MEMORY, prioritized_replay):
    # initial variable caching, done only once
    if prioritized_replay:
        # initialize prioritized replay buffer
        t, D = 0, PrioritizedReplayBuffer(maxlen=REPLAY_MEMORY)
    else:
        # initialize standard replay buffer
        t, D = 0, ReplayBuffer(maxlen=REPLAY_MEMORY)
    
     # set up dictionary to store initial settings
    set_up_dict = {"epsilon": INITIAL_EPSILON, "step": t, "D": D, "highest_score": 0}
    save_obj(set_up_dict, "set_up") # save initial settings

def save_obj(obj, name):
    # save an object to a file using pickle
    with open('./result/'+ name + '.pkl', 'wb') as f: 
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    # load an object from a file using pickle
    with open('result/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
