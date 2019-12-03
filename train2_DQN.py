"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import argparse
import wimblepong
from DQN_SAA.DQN_SAA import *
from utils import *
import pickle
import os

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--glie_a", type=int, help="GLIE-a value", default=1000)
parser.add_argument("--size", type=int, default=120)
parser.add_argument("--fc1_size", type=int, default=64)
parser.add_argument("--target_update", type=int, default=10)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("--load", action="store_true")
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 500000

# Define the player IDs for both SimpleAI agents
player1_id = 1
player2_id = 3 - player1_id

with open("DQN_SAA/bigger-CNN/model_info.p", "rb") as f:
    model_info = pickle.load(f)

model_info["model_name"] = "two_agents"
with open("DQN_SAA/two_agents/model_info.p", "wb") as f:
    pickle.dump(model_info, f)

player1 = DQN_SAA(env, player1_id, model_info=model_info, fc1_size=args.fc1_size)
player2 = DQN_SAA(env, player2_id, model_info=model_info, fc1_size=args.fc1_size)

glie_a = args.glie_a
target_update = args.target_update

# Set the names for both SimpleAIs
env.set_names(player1.get_name(), player2.get_name())

win1 = 0
wins = []
wins2 = []
avg_over = 50

RA_actions = 0
rewards = []
for i in range(1, episodes):
    done = False

    state1, state2 = env.reset()

    state1 = process_state(state1, player1.size)
    state2 = process_state(state2, player2.size)
    state_diff1 = 2 * state1 - state1
    state_diff2 = 2 * state2 - state2

    actions = 0
    while not done:
        actions += 1
        # Get the actions from both SimpleAIs
        action1 = player1.get_action(state_diff1)
        action2 = player2.get_action(state_diff2)
        # Step the environment and get the rewards and new observations
        (next_state1, next_state2), (rew1, rew2), done, info = env.step((action1, action2))

        next_state1 = process_state(next_state1, player1.size)
        next_state2 = process_state(next_state2, player2.size)

        next_state_diff1 = 2 * next_state1 - state1
        next_state_diff2 = 2 * next_state2 - state2

        player1.store_transition(state_diff1, action1, next_state_diff1, rew1, done)
        player2.store_transition(state_diff2, action2, next_state_diff2, rew2, done)

        state_diff1 = next_state_diff1
        state_diff2 = next_state_diff2
        state1 = next_state1
        state2 = next_state2

        if rew1 == 10:
            win1 += 1

        if done:
            rewards.append(rew1)

            player1.update_network()
            player2.update_network()

            if rew1 == 10:
                wins.append(1)
            else:
                wins.append(0)

            if rew2 == 10:
                wins2.append(1)
            else:
                wins2.append(0)

            if len(wins) > avg_over:
                wins = wins[-avg_over:]

            if len(wins2) > avg_over:
                wins2 = wins2[-avg_over:]

            if i % target_update == 0:
                player1.update_target_network()
                player2.update_target_network()
                print("Target network updated!")

            observation = env.reset()
            RA_actions = 0.9 * RA_actions + 0.1 * actions
            print("episode {} over. RWR1: {:.3f}. RWR2 {:.3f}. RAA: {:.3f}.".format(i, np.mean(wins), np.mean(wins2), RA_actions))

    if i % 100 == 0:
        chosen_actions = player1.chosen_actions
        if np.sum(chosen_actions) != 0:
            chosen_actions /= np.sum(chosen_actions)

        chosen_actions = np.round(chosen_actions, 2)
        print("Action distribution:", list(chosen_actions))

        if args.save:
            with open("DQN_SAA/two_agents/rewards.p", "wb") as f:
                pickle.dump(rewards, f)

            if np.mean(wins) > np.mean(wins2):
                torch.save(player1.policy_net, "DQN_SAA/two_agents/policy_net.pth")
                player2.policy_net = player1.policy_net
                print("Model 1 saved!")
            else:
                torch.save(player2.policy_net, "DQN_SAA/two_agents/policy_net.pth")
                player1.policy_net = player2.policy_net
                print("Model 2 saved!")

            with open("DQN_SAA/two_agents/performance.txt", "a") as f:
                f.write("episode {} over. RWR1: {:.3f}. RWR2 {:.3f}. RAA: {:.3f}.".format(i, np.mean(wins), np.mean(wins2), RA_actions))
                f.write("\n")