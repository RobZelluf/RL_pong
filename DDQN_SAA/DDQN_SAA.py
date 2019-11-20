from wimblepong import Wimblepong
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import Transition, ReplayMemory


class Q_CNN(nn.Module):
    def __init__(self, state_space, action_space, size):
        super(Q_CNN, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.linear_size = int(((size - 8) / 2)**2 * 8)

        self.conv1 = nn.Conv2d(1, 4, 4, 2)
        self.conv2 = nn.Conv2d(4, 8, 4, 1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(self.linear_size, 64)
        self.fc2 = torch.nn.Linear(64, action_space)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, self.linear_size)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return x


class DDQN_SAA(object):
    def __init__(self, env, player_id=1, load=False, replay_buffer_size=50000, batch_size=256, gamma=0.98, size=200):
        if type(env) is not Wimblepong:
            raise TypeError("I'm not a very smart AI. All I can play is Wimblepong.")

        self.env = env
        self.player_id = player_id
        self.name = "SAA"
        self.gamma = gamma
        self.size = size

        if torch.cuda.is_available():
            print("Using GPU!")
            torch.cuda.set_device(0)

        self.state_space = env.observation_space
        self.action_space = env.action_space.n
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.chosen_actions = np.zeros(self.action_space)

        if load:
            self.network1 = torch.load("DDQN_SAA/policy_net.pth")
            self.network2 = torch.load("DDQN_SAA/target_net.pth")
            print("Policy and target loaded!")
        else:
            self.network1 = Q_CNN(self.state_space, self.action_space, size)
            self.network2 = Q_CNN(self.state_space, self.action_space, size)

        self.optimizer1 = optim.RMSprop(self.network1.parameters(), lr=1e-3)
        self.optimizer2 = optim.RMSprop(self.network2.parameters(), lr=1e-3)

    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = 1 - torch.tensor(batch.done, dtype=torch.uint8)
        non_final_next_states = [s for nonfinal, s in zip(non_final_mask,
                                                          batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states)
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        loss1, loss2 = self.get_losses(action_batch, non_final_mask, non_final_next_states, reward_batch, state_batch)

        # Optimize the model
        self.optimizer1.zero_grad()
        loss1.backward()
        for param in self.network1.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer1.step()

        # Optimize the model
        self.optimizer1.zero_grad()
        loss2.backward()
        for param in self.network2.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer2.step()

    def get_losses(self, action_batch, non_final_mask, non_final_next_states, reward_batch, state_batch):
        # Update model 1
        state_action_values1 = self.network1(state_batch).gather(1, action_batch)
        state_action_values2 = self.network2(state_batch).gather(1, action_batch)

        next_state_values1 = torch.zeros(self.batch_size)
        next_state_values2 = torch.zeros(self.batch_size)
        next_state_values1[non_final_mask] = self.network1(non_final_next_states).max(1)[0].detach()
        next_state_values2[non_final_mask] = self.network2(non_final_next_states).max(1)[0].detach()
        next_state_values = torch.min(next_state_values1, next_state_values2)

        expected_state_action_values = reward_batch + self.gamma * next_state_values
        # Compute Huber loss
        loss1 = F.smooth_l1_loss(state_action_values1.squeeze(),
                                 expected_state_action_values)

        loss2 = F.smooth_l1_loss(state_action_values2.squeeze(),
                                 expected_state_action_values)

        return loss1, loss2

    def get_name(self):
        return self.name

    def get_action(self, state, epsilon=0.05):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = state.reshape(1, 1, self.size, self.size)
                state = torch.from_numpy(state).float()
                q_values = self.network1(state)
                action = torch.argmax(q_values).item()
                self.chosen_actions[action] += 1

                return action
        else:
            return random.randrange(self.action_space)

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(next_state).float()
        state = torch.from_numpy(state).float()
        self.memory.push(state, action, next_state, reward, done)