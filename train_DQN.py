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

if args.save and not args.load:
    model_name = input("Model name/number:")
    if not os.path.exists("DQN_SAA/" + model_name):
        os.mkdir("DQN_SAA/" + model_name)

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

if args.load:
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
        if "fc1_size" in model_info:
            fc1_size = model_info["fc1_size"]
        else:
            fc1_size = 64

    player = DQN_SAA(env, player_id, model_info=model_info, fc1_size=fc1_size)
else:
    player = DQN_SAA(env, player_id, size=args.size, fc1_size=args.fc1_size)
    start_episode = 0

glie_a = args.glie_a
target_update = args.target_update

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

win1 = 0
wins = []
avg_over = 50

RA_actions = 0
for i in range(start_episode, episodes):
    done = False
    if glie_a <= 0:
        eps = 0
    else:
        eps = glie_a / (glie_a + i)
        if eps < 0.5:
            if random.random() > 0.5:
                eps = 0.5
            else:
                eps = 0

    state, _ = env.reset()
    state = process_state(state, player.size)
    state_diff = 2 * state - state

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

        if rew1 == 10:
            win1 += 1
        else:
            rew1 = 0.01

        player.store_transition(state_diff, action1, next_state_diff, rew1, done)

        state_diff = next_state_diff
        state = next_state

        if done:
            player.update_network()

            if rew1 == 10:
                wins.append(1)
            else:
                wins.append(0)

            if len(wins) > avg_over:
                wins = wins[-avg_over:]

            if i % target_update == 0:
                player.update_target_network()
                print("Target network updated!")

            observation = env.reset()
            RA_actions = 0.9 * RA_actions + 0.1 * actions
            print("episode {} over. RWR: {:.3f}. RAA: {:.3f}. Ep: {:.3f}".format(i, np.mean(wins), RA_actions, eps))

    if i % 100 == 0:
        chosen_actions = player.chosen_actions
        if np.sum(chosen_actions) != 0:
            chosen_actions /= np.sum(chosen_actions)

        chosen_actions = np.round(chosen_actions, 2)
        print("Action distribution:", list(chosen_actions))

        if args.save:
            torch.save(player.policy_net, "DQN_SAA/" + model_name + "/policy_net.pth")
            model_info = dict()
            model_info["model_name"] = model_name
            model_info["size"] = player.size
            model_info["episode"] = i
            model_info["fc1_size"] = player.fc1_size

            with open("DQN_SAA/" + model_name + "/model_info.p", "wb") as f:
                pickle.dump(model_info, f)

            print("Model", model_name,"saved!")

            with open("DQN_SAA/" + model_name + "/performance.txt", "a") as f:
                f.write("episode {} over. RWR: {:.3f}. RAA: {:.3f}. Ep: {:.3f}".format(i, np.mean(wins), RA_actions, eps))
                f.write("\n")