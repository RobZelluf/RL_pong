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
parser.add_argument("--fps", type=int, help="FPS for rendering", default=2000)
parser.add_argument("--glie_a", type=int, help="GLIE-a value", default=30)
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
for i in range(start_episode, episodes):
    done = False
    eps = glie_a / (glie_a + i)

    state, _ = env.reset()
    state = np.transpose(state)
    state_diff = state - state
    point = 0
    while not done:
        # Get the actions from both SimpleAIs
        action1 = player.get_action(state, eps)
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (next_state, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        if not i % 10 == 0:
            next_state = np.transpose(next_state)
            next_state_diff = next_state - state

            player.store_transition(state_diff, action1, next_state_diff, rew1, done)
            player.update_network()
            state_diff = next_state_diff

        #img = Image.fromarray(ob1)
        #img.save("ob1.png")
        #img = Image.fromarray(ob2)
        #img.save("ob2.png")
        # Count the wins
        if rew1 == 10:
            win1 += 1
            point = 1
        if i % 10 == 0 and not args.headless:
            env.render()
        if done:
            observation= env.reset()
            print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
            print("Epsilon:", eps)

    cumulative_rewards.append(0.9 * cumulative_rewards[-1] + 0.1 * point)
    print("Last average reward:", cumulative_rewards[-1])
    if not args.headless:
        plot_rewards(cumulative_rewards)

    if i % 10 == 0:
        print("Models saved!")
        torch.save(player.policy_net, "models/policy_net.pth")
        torch.save(player.target_net, "models/target_net.pth")
        with open("models/model_info.p", "wb") as f:
            pickle.dump(i, f)
