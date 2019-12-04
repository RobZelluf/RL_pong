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
import os

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=60)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 100000

DIRs = [x for x in os.listdir("DQN_SAA/") if os.path.isdir("DQN_SAA/" + x) and "cache" not in x]
i = 0
for DIR in DIRs:
    print(i, DIR)
    i += 1

model_ind = int(input("Model number:"))
model_name = DIRs[model_ind]

with open("DQN_SAA/" + model_name + "/model_info.p", "rb") as f:
    model_info = pickle.load(f)

start_episode = model_info["episode"]

ignore_opponent = False
if "ignore_opponent" in model_info:
    ignore_opponent = model_info["ignore_opponent"]

print("Ignoring opponent:", ignore_opponent)

# Define the player IDs for both SimpleAI agents
player1_id = 1
player2_id = 3 - player1_id

player1 = DQN_SAA(env, player1_id, model_info=model_info)
player2 = DQN_SAA(env, player2_id, model_info=model_info)

# Set the names for both SimpleAIs
env.set_names(player1.get_name(), player2.get_name())

win1 = 0
for i in range(0, episodes):
    done = False

    state1, state2 = env.reset()

    state1 = process_state(state1, player1.size, ignore_opponent)
    state2 = process_state(state2, player2.size, ignore_opponent)
    state_diff1 = 2 * state1 - state1
    state_diff2 = 2 * state2 - state2

    actions = 0
    while not done:
        actions += 1
        # Get the actions from both SimpleAIs
        action1 = player1.get_action(state_diff1, 0)
        action2 = player2.get_action(state_diff2, 0)
        # Step the environment and get the rewards and new observations
        (next_state1, next_state2), (rew1, rew2), done, info = env.step((action1, action2))

        next_state1 = process_state(next_state1, player1.size, ignore_opponent)
        next_state2 = process_state(next_state2, player2.size, ignore_opponent)

        next_state_diff1 = 2 * next_state1 - state1
        next_state_diff2 = 2 * next_state2 - state2

        state_diff1 = next_state_diff1
        state_diff2 = next_state_diff2
        state1 = next_state1
        state2 = next_state2

        env.render()

        if rew1 == 10:
            win1 += 1
            point = 1

        if done:
            env.reset()
