from wimblepong import Wimblepong
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(nn.Module):
    def __init__(self, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space

        self.conv1 = nn.Conv2d(1, 4, 1, 1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(100 * 100 * 4, 64)
        self.fc2_mean = torch.nn.Linear(64, action_space)
        self.fc2_value = torch.nn.Linear(64, 1)

        self.sigma0 = self.sigma = 2
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, k):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 100 * 100 * 4)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)

        mu = self.fc2_mean(x)
        self.sigma = np.sqrt(self.sigma0 * np.exp(-0.00005 * k))

        dist = torch.distributions.Normal(mu, self.sigma)
        state_value = self.fc2_value(x)[0]
        return dist, state_value


class AC_SAA(object):
    def __init__(self, env, policy, player_id=1, load=False, replay_buffer_size=500, batch_size=500, gamma=0.98):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")

        self.env = env
        self.player_id = player_id
        self.name = "SAA"

        if torch.cuda.is_available():
            print("Using GPU!")
            torch.cuda.set_device(0)

        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.value_estimates = []
        self.episode_number = 0

    def episode_finished(self, episode_number):
        self.episode_number = episode_number
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)

        state_values = torch.stack(self.value_estimates, dim=0).to(self.train_device).squeeze(-1)
        next_state_values = torch.cat((state_values[1:], torch.zeros(1)))

        self.states, self.action_probs, self.rewards, self.value_estimates = [], [], [], []
        # self.sigma = np.sqrt(5 * np.exp(-5*10**-4 * episode_number))

        # TODO: Compute discounted rewards (use the discount_rewards function)
        discounted_rewards = discount_rewards(rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # TODO: Compute critic loss and advantages (T3)
        advantages = discounted_rewards + self.gamma * next_state_values - state_values
        critic_loss = torch.mean((discounted_rewards - state_values)**2)

        ad = advantages.detach()
        loss = action_probs * ad

        # TODO: Compute the optimization term (T1, T3)
        loss = torch.mean(-loss) + critic_loss

        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()

        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_name(self):
        return self.name

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        x = x.reshape(1, 1, 200, 200)

        # TODO: Pass state x through the policy network (T1)
        dist, state_value = self.policy.forward(x, self.episode_number)

        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = dist.mean
        else:
            action = dist.sample()

        if abs(action) <= 0.5:
            action = self.env.STAY
        elif action > 0.5:
            action = self.env.MOVE_UP
        else:
            action = self.env.MOVE_DOWN

        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = dist.log_prob(action)

        # TODO: Return state value prediction, and/or save it somewhere (T3)
        return action, act_log_prob, state_value

    def store_outcome(self, observation, action_prob, action_taken, reward, value_est):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.value_estimates.append(value_est)
