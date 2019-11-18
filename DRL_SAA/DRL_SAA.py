from wimblepong import Wimblepong
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from DRL_SAA.utils import *


class Policy(nn.Module):
    def __init__(self, state_space, action_space, hidden_layer=64):
        super(Policy, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.fc1 = torch.nn.Linear(state_space, hidden_layer)
        self.fc2 = torch.nn.Linear(hidden_layer, action_space)

    def forward(self, x):
        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = F.softmax(self.fc2(x))
        return -torch.log(x)


class DRL_SAA(object):
    def __init__(self, env, player_id=1):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")

        self.env = env
        self.player_id = player_id
        self.name = "SAA"
        self.gamma = 0.98

        self.states, self.action_probs, self.rewards = [], [], []

        if torch.cuda.is_available():
            print("Using GPU!")
            torch.cuda.set_device(0)

        self.state_space = 200 * 200
        self.action_space = env.action_space.n

        self.policy_net = Policy(self.state_space, self.action_space)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-3)

    def get_name(self):
        return self.name

    def get_action(self, state, epsilon=0.0):
        sample = random.random()

        with torch.no_grad():
            state = torch.from_numpy(state).float()
            q_values = self.policy_net(state)

            if sample > epsilon:
                return torch.argmax(q_values).item(), torch.max(q_values)
            else:
                action = random.randrange(self.action_space)
                print("LOL")
                return action, q_values[action]

    def update_network(self):
        action_probs = torch.stack(self.action_probs, dim=0).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).squeeze(-1)

        self.states, self.action_probs, self.rewards = [], [], []

        discounted_rewards = discount_rewards(rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        loss = action_probs * discounted_rewards
        loss = torch.mean(-loss)

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

    def store_outcome(self, observation, action_prob, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

