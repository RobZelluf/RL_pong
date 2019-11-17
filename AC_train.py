"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from PIL import Image
from AC_SAA.AC_SAA import *
from utils import *
import pickle

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--save", action="store_true")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--glie_a", type=int, help="GLIE-a value", default=500)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("--load", action="store_true")
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 100000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)

action_space_dim = 1

policy = Policy(action_space_dim)
player = SAA(env, policy)

start_episode = 0

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

win1 = 0
cumulative_rewards = [0]
RA_actions = [0]
for i in range(start_episode, episodes):
    state, _ = env.reset()
    state = process_state(state)
    state_diff = state - state

    done = False

    actions, won = 0, 0
    while not done:
        actions += 1
        # Get action from the agent
        action1, action_probabilities, state_value = player.get_action(state_diff)
        action2 = opponent.get_action()

        previous_state_diff = state_diff

        # Perform the action on the environment, get new state and reward
        (next_state, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        if rew1 == 0:
            rew1 = 0.01

        next_state = process_state(next_state)
        state_diff = next_state - state
        state = next_state

        # Store action's outcome (so that the agent can improve its policy)
        player.store_outcome(previous_state_diff, action_probabilities, action1, rew1, state_value)

        if rew1 == 10:
            win1 += 1
            won = 1

    player.episode_finished(i)

    cumulative_rewards.append(0.9 * cumulative_rewards[-1] + 0.1 * won)
    RA_actions.append(0.9 * RA_actions[-1] + 0.1 * actions)
    print("episode {} over. Broken WR: {:.3f}. LAR: {:.3f}. RAA: {:.3f}".format(i, win1 / (i + 1), cumulative_rewards[-1],
                                                                              RA_actions[-1]))
