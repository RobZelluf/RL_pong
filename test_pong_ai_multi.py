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
from DQN_SAA.DQN_SAA import *
from utils import process_state

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
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
player = DQN_SAA(env, player_id, True)

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

win1 = 0
for i in range(0,episodes):
    done = False

    state, _ = env.reset()
    state = process_state(state)
    state_diff = 2 * state - state

    actions = 0
    while not done:
        actions += 1
        # Get the actions from both SimpleAIs
        action1 = player.get_action(state_diff, 0)
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (next_state, ob2), (rew1, rew2), done, info = env.step((action1, action2))

        next_state = process_state(next_state)
        next_state_diff = 2 * next_state - state

        player.store_transition(state_diff, action1, next_state_diff, rew1, done)

        state_diff = next_state_diff
        state = next_state

        env.render()

        if rew1 == 10:
            win1 += 1
            point = 1

        if done:
            env.reset()
