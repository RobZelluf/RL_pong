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
from superawesomeagent.superawesomeagent import *
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
player = SAA(env, player_id)
start_episode = 0

if args.load:
    player.target_net = torch.load("models/target_net.pth")
    player.policy_net = torch.load("models/policy_net.pth")
    with open("models/model_info.p", "rb") as f:
        start_episode = pickle.load(f)

glie_a = args.glie_a

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

win1 = 0
cumulative_rewards = [0]
RA_actions = [0]
for i in range(start_episode, episodes):
    done = False
    eps = glie_a / (glie_a + i)

    state, _ = env.reset()
    state = process_state(state)
    state_diff = 2 * state - state
    point = 0

    actions = 0
    while not done:
        actions += 1
        # Get the actions from both SimpleAIs
        action1 = player.get_action(state_diff, eps)
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (next_state, ob2), (rew1, rew2), done, info = env.step((action1, action2))

        next_state = process_state(next_state)
        next_state_diff = 2 * next_state - state

        player.store_transition(state_diff, action1, next_state_diff, rew1, done)

        state_diff = next_state_diff
        state = next_state

        #img = Image.fromarray(ob1)
        #img.save("ob1.png")
        #img = Image.fromarray(ob2)
        #img.save("ob2.png")
        # Count the wins

        if rew1 == 10:
            win1 += 1
            point = 1

        if done:
            player.update_network()
            observation = env.reset()
            cumulative_rewards.append(0.9 * cumulative_rewards[-1] + 0.1 * point)
            RA_actions.append(0.9 * RA_actions[-1] + 0.1 * actions)
            print("episode {} over. Broken WR: {:.3f}. LAR: {:.3f}. RAA: {:.3f}".format(i, win1/(i+1), cumulative_rewards[-1], RA_actions[-1]))
            print("Epsilon: {:.3f}".format(eps))

    if not args.headless:
        plot_rewards(cumulative_rewards)

    if i % 100 == 0 and args.save:
        torch.save(player.policy_net, "models/policy_net.pth")
        torch.save(player.target_net, "models/target_net.pth")
        with open("models/model_info.p", "wb") as f:
            pickle.dump(i, f)

        print("Models saved!")
