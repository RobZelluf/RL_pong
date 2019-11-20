import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import random
import cv2


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done', 'actions'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        p = list(np.array([x.actions for x in self.memory]) + 50)
        p /= np.sum(p)
        mask = np.random.choice(len(self.memory), batch_size, replace=False, p=p)
        return [self.memory[i] for i in mask]

    def __len__(self):
        return len(self.memory)


def plot_rewards(rewards):
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.grid(True)
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=32):
        super(DQN, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(state_space_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_space_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def process_state(state, size):
    mask = np.all(state == [43, 48, 58], axis=-1)
    state[mask] = [0, 0, 0]
    state = np.mean(state, axis=-1)
    state = cv2.resize(state, (size, size))
    state = state.astype(int)
    state = state.reshape((1, size, size))
    return state

