from wimblepong import Wimblepong
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import Transition, ReplayMemory
import pickle
import cv2


class Q_CNN(nn.Module):
    def __init__(self, state_space, action_space, size, fc1_size=64):
        super(Q_CNN, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.linear_size = int((size / 2 - 8)**2 * 4)

        self.conv1 = nn.Conv2d(1, 16, 8, 2)
        self.conv2 = nn.Conv2d(16, 8, 4, 1)
        self.conv3 = nn.Conv2d(8, 4, 3, 1)
        self.fc1 = torch.nn.Linear(self.linear_size, fc1_size)
        self.fc2 = torch.nn.Linear(fc1_size, action_space)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, self.fc1.in_features)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return x


class Agent(object):
    def __init__(self, player_id=1, size=120, fc1_size=64):
        self.player_id = player_id
        self.name = "SAA-BN2v2"
        self.size = size
        self.fc1_size = fc1_size
        self.model_info = None
        self.prev_state = None
        self.ignore_opponent = False

        if torch.cuda.is_available():
            print("Using GPU!")
            torch.cuda.set_device(0)

        self.memory = ReplayMemory()
        self.batch_size = 256 * 2

    def load_model(self):
        self.policy_net = torch.load("policy_net.pth")
        with open("model_info.p", "rb") as f:
            self.model_info = pickle.load(f)

        if 'ignore_opponent' in self.model_info:
            self.ignore_opponent = self.model_info["ignore_opponent"]

    def reset(self):
        self.prev_state = None

    def get_name(self):
        return self.name

    def get_action(self, state):
        if self.prev_state is None:
            self.prev_state = process_state(state, self.size, self.ignore_opponent)
            return 0
        else:
            state = process_state(state, self.size, self.ignore_opponent)
            state_diff = get_state_diff(state, self.prev_state)
            self.prev_state = state

            with torch.no_grad():
                state_diff = state_diff.reshape(1, 1, self.size, self.size)
                state_diff = torch.from_numpy(state_diff).float()
                q_values = self.policy_net(state_diff)
                action = torch.argmax(q_values).item()

                return action


def get_state_diff(state, prev_state):
    return 2 * state - prev_state

def process_state(state, size, ignore_opponent=False):
    mask = np.all(state == [43, 48, 58], axis=-1)
    state[mask] = [0, 0, 0]
    state = np.mean(state, axis=-1)

    if ignore_opponent:
        state[:, 180:] = 0

    if size < 200:
        state = cv2.resize(state, (size, size))
    state = state.astype(int)
    state = state.reshape((1, size, size))
    return state
