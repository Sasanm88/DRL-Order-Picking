import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from collections import namedtuple, deque
from environment import *
import torch.optim as optim

import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition',
                        ('stateM', 'stateP', 'action', 'next_stateM', 'next_stateP', 'reward', 'allowed_actions'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class QNetwork(nn.Module):
    def __init__(self, action_size, n_aisle):
        super(QNetwork, self).__init__()
        self.fcl1 = nn.Linear(4, 64)
        self.fcl2 = nn.Linear(2*n_aisle, n_aisle*16)
        self.fcl3 = nn.Linear((n_aisle+4)*16, 256)
        self.fcl4 = nn.Linear(256, 128)
        self.fcl6 = nn.Linear(128, 64)
        self.fcl5 = nn.Linear(64, action_size)

    def forward(self, x1, x2):
        x1 = torch.relu(self.fcl1(x1))
        x2 = torch.relu(self.fcl2(x2))
        x3 = torch.cat((x1, x2), 1)
        x4 = torch.relu(self.fcl3(x3))
        x4 = torch.relu(self.fcl4(x4))
        x4 = torch.relu(self.fcl6(x4))
        x5 = self.fcl5(x4)
        return x5

    
    def select_action(self, stateM, stateP, steps_done, allowed_actions, EPS_START, EPS_END, EPS_DECAY):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)

        random_selection = sample < eps_threshold

        if random_selection: 
            return random.choice(allowed_actions) # allowed_actions[torch.randint(0,allowed_actions.shape[0], (1,)).item()]
        else:
            self.eval()
            with torch.no_grad():
                return allowed_actions[torch.argmax(self.forward(stateM.unsqueeze(0), stateP.unsqueeze(0))[0][allowed_actions])]

            self.train()
    def select_optimal_action(self, stateM, stateP, allowed_actions):
        self.eval()
        with torch.no_grad():
            return allowed_actions[torch.argmax(self.forward(stateM.unsqueeze(0), stateP.unsqueeze(0))[0][allowed_actions])]

class Agent(): 
    def __init__(self, n_row, n_aisle, action_size, arrival_rate, capacity, depot, inter_aisle_distance, pick_up_time, drop_off_time, alpha, LR, memory_size, batch_size, Gamma, tau, EPS_START, EPS_END, EPS_DECAY, lr_decrease_step):
        self.n_row = n_row
        self.n_aisle = n_aisle
        self.n_col = n_aisle*3
        self.action_size = action_size
        self.arrival_rate = arrival_rate
        self.capacity = capacity
        self.depot = depot
        self.inter_aisle_distance = inter_aisle_distance
        self.pick_up_time = pick_up_time
        self.drop_off_time = drop_off_time
        self.alpha = alpha
        
        self.batch_size = batch_size
        self.gamma = Gamma
        self.tau = tau 
        self.ep1 = EPS_START
        self.ep2 = EPS_END
        self.epd = EPS_DECAY
        self.lr = LR
        self.lr_decrease_step = lr_decrease_step
        
        self.env = Env(self.n_row, self.n_aisle, self.arrival_rate, self.capacity, depot, inter_aisle_distance, pick_up_time, drop_off_time, alpha)
        self.policy_net = QNetwork(action_size, self.n_aisle).to(device)
        self.target_net = QNetwork(action_size, self.n_aisle).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR)
        
        self.replay = ReplayMemory(memory_size)
        self.t_step = 0
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        
    def select_action(self, stateM, stateP, allowed_actions):
        return self.policy_net.select_action(stateM, stateP, self.t_step, allowed_actions, self.ep1, self.ep2, self.epd)
        
    def step(self, stateM, stateP, action, reward, next_stateM, next_stateP, allowed_actions_tensor, step_num):
        self.replay.push(stateM, stateP, action, next_stateM, next_stateP, reward, allowed_actions_tensor)
        
        if self.replay.__len__() > self.batch_size:
            transitions = self.replay.sample(self.batch_size)
            self.learn(transitions, step_num)
            
    def learn(self, transitions, step_num):
        batch = Transition(*zip(*transitions))

        stateM_batch = torch.stack(batch.stateM, dim=0)
        next_stateM_batch = torch.stack(batch.next_stateM, dim=0)
        stateP_batch = torch.stack(batch.stateP, dim=0)
        next_stateP_batch = torch.stack(batch.next_stateP, dim=0)

        action_batch = torch.tensor(batch.action, device = device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device = device)
        allowed_actions_batch = torch.stack(batch.allowed_actions, dim=0)

        self.policy_net.train()
        self.target_net.eval()

        state_action_values = self.policy_net(stateM_batch, stateP_batch).gather(1, action_batch)

        with torch.no_grad():
            x = self.target_net(next_stateM_batch, next_stateP_batch)
        next_state_values = (allowed_actions_batch * x).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        if step_num%10 == 0:
            self.network_update()
        
    def network_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
        
    def train(self, n_episodes, max_t):
        scores = []
        MA = []
        best_score = torch.tensor(-np.inf, device = device)
        scores_window = deque(maxlen=50)
        start_time = time.time()
        lr_decrease_num = 1
        for episode in range(n_episodes):
            
            if (episode+1) == round(lr_decrease_num*n_episodes/(self.lr_decrease_step+1)) == 0:
                lr_decrease_num += 1
                self.lr = self.lr/2
                self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr)
            
            stateM, stateP, allowed_actions, allowed_actions_tensor = self.env.reset()
            score = torch.tensor(0.0, device = device)
            for t in range(max_t):
                old_stateM = stateM
                old_stateP = stateP
                action = self.select_action(stateM, stateP, allowed_actions)
#                 print(action.device)
                stateM, stateP, reward, allowed_actions, allowed_actions_tensor = self.env.step(action)

                score = score + reward
                new_stateM = stateM
                new_stateP = stateP
                self.step(old_stateM, old_stateP, action, reward, new_stateM, new_stateP, allowed_actions_tensor, t+1)
                
            self.t_step += 1
            scores.append(score)
            if len(MA) == 0:
                MA.append(score)
            else: 
                MA.append(0.01*score + 0.99*MA[-1])
            scores_window.append(score)
            time_spent = time.time() - start_time
            if episode%1==0:
                best_score = self.save_best_network(best_score)
#             mean_score = torch.mean(scores_window)
            self.update_plot(episode, MA, scores_window, time_spent, best_score)
            plt.close()
#             print(f'Episode {episode}  Average Score {mean_score:.2f} (training time: {time_spent:.1f} seconds)', end='\r')
            
            
#             if score > best_score: 
#                 best_score = score
#                 torch.save(self.target_net.state_dict(), 'saved_models/check_point_nrow'+str(self.n_row)+'_ncol'+str(self.n_col)+'.pth')
        return scores
    
    def update_plot(self, epoch, losses, scores_window, time_spent, best_score):
        plt.figure(figsize=(8, 4))
        plt.plot(losses, marker='o', linestyle='-')
        plt.title(f'Score Progress (Episode {epoch}) - (Average score last 50: {np.mean(scores_window):.2f}) - (Best Score: {best_score}) - (training time: {time_spent:.1f} seconds)')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid()
        display.clear_output(wait=True)
        display.display(plt.gcf())

    def save_best_network(self, best_score):
        env = Env(self.n_row, self.n_aisle, self.arrival_rate, self.capacity, self.depot, self.inter_aisle_distance, self.pick_up_time, self.drop_off_time, self.alpha)
        stateM, stateP, allowed_actions, _ = env.reset()
        score = 0
        with torch.no_grad():
            for i in range((self.n_aisle)*100):
                action = self.target_net.select_optimal_action(stateM, stateP, allowed_actions)
                stateM, stateP, reward, allowed_actions, _ = env.step(action)
                score += reward
        if score > best_score:
            best_score = score.item()
            torch.save(self.target_net.state_dict(), 'saved_models/check_point_arrival_rate'+str(self.arrival_rate)+'_alpha'+str(self.alpha)+'.pth')
        return best_score