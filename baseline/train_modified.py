############ import libraries ############

from misc.utils import AverageMeter, save_obj
import numpy as np
import torch
import pandas as pd
import os
import sys
sys.path.append("../")

#########################################
class trainNetwork:
    def __init__(self, agent, game, device):
        """initialize the training process with the given agent, game, and device"""
        self.agent = agent
        self.game = game
        self.device = device
        self.episode_rewards = []  # store rewards for each episode

        # ensure results directory exists
        os.makedirs('./results', exist_ok=True)
        os.makedirs('./weights', exist_ok=True)

    def save(self):
        """save the episode rewards to disk"""
        np.save('./results/episode_rewards.npy', self.episode_rewards)

        df = pd.DataFrame({'episode': list(range(len(self.episode_rewards))), 'reward': self.episode_rewards})
        df.to_csv('./results/episode_rewards.csv', index=False)

    def start(self, EPISODE, ACTIONS, FRAME_PER_ACTION):
        current_episode = 0
        
        while current_episode < EPISODE:
            step = 0
            self.game.restart()  # make sure to restart the game for each episode
            x_t, _, terminal = self.game.get_state(np.zeros(ACTIONS))  # initial state
            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # stack 4 images to create placeholder input
            s_t = s_t.reshape(1, s_t.shape[2], s_t.shape[0], s_t.shape[1])  # 1*4*80*80
            s_t = torch.from_numpy(s_t).float()
            
            while not self.game.get_crashed():
                action_idx = 1  # always jump
                a_t = np.zeros([ACTIONS])
                a_t[action_idx] = 1

                x_t1, _, terminal = self.game.get_state(a_t)
                x_t1 = x_t1[:, :, np.newaxis]
                x_t1 = x_t1.reshape((1, 1, 80, 80))
                s_t1 = np.concatenate((x_t1, s_t[:, :3, :, :]), axis=1)
                s_t1 = torch.from_numpy(s_t1).float()

                s_t = s_t1
                step += 1
                if terminal:
                    break

            # get the score from the game as the total reward
            total_reward = self.game.get_score()
            self.episode_rewards.append(total_reward)
            current_episode += 1
            print(f"Episode: {current_episode}, Total Reward: {total_reward}")

        self.save()  # save results at the end
