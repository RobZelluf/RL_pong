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
episodes = 500000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)
player = DQN_SAA(env, player_id, size=80)
start_episode = 0

if args.load:
    player.policy_net = torch.load("DQN_SAA/policy_net.pth")
    player.update_target_network()
    with open("DQN_SAA/model_info.p", "rb") as f:
        start_episode = pickle.load(f)

glie_a = args.glie_a

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

win1 = 0
cumulative_rewards = [0]
RA_actions = [0]
for i in range(start_episode, episodes):
    done = False
    if glie_a <= 0:
        eps = 0
    else:
        eps = glie_a / (glie_a + i)
        eps = max(0.05, eps)

    state, _ = env.reset()
    state = process_state(state, player.size)
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

        next_state = process_state(next_state, player.size)
        next_state_diff = 2 * next_state - state

        player.store_transition(state_diff, action1, next_state_diff, rew1, done)

        state_diff = next_state_diff
        state = next_state

        if rew1 == 10:
            win1 += 1
            point = 1

        if done:
            player.update_network()

            if i % 20 == 0:
                player.update_target_network()
                print("Target network updated!")

            observation = env.reset()
            cumulative_rewards.append(0.9 * cumulative_rewards[-1] + 0.1 * point)
            RA_actions.append(0.9 * RA_actions[-1] + 0.1 * actions)
            print("episode {} over. Broken WR: {:.3f}. LAR: {:.3f}. RAA: {:.3f}. Ep: {:.3f}".format(i, win1/(i+1), cumulative_rewards[-1], RA_actions[-1], eps))

    if i % 100 == 0:
        chosen_actions = player.chosen_actions
        if np.sum(chosen_actions) != 0:
            chosen_actions /= np.sum(chosen_actions)

        chosen_actions = np.round(chosen_actions, 2)
        print("Action distribution:", list(chosen_actions))

        if args.save:
            torch.save(player.policy_net, "DQN_SAA/policy_net.pth")
            with open("DQN_SAA/model_info.p", "wb") as f:
                pickle.dump(i, f)

            print("Models saved!")
