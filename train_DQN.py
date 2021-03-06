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
test_episodes = 100
failed = 0
restart_after_fails = 5
ignore_opponent = False
model_info = dict()

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
        start_episode = model_info["episode"] + 1
        if "fc1_size" in model_info:
            fc1_size = model_info["fc1_size"]
        else:
            fc1_size = 64

        if "ignore_opponent" in model_info:
            ignore_opponent = model_info["ignore_opponent"]
        else:
            if input("Ignore opponent? (y/n)") == "y":
                ignore_opponent = True

    player = DQN_SAA(env, player_id, model_info=model_info, fc1_size=fc1_size)
else:
    if input("Ignore opponent? (y/n)") == "y":
        ignore_opponent = True
    player = DQN_SAA(env, player_id, size=args.size, fc1_size=args.fc1_size)
    start_episode = 1

if args.save:
    with open("DQN_SAA/" + model_name + "/model_info.txt", "w") as f:
        f.write(str(player.policy_net.conv1) + "\n")
        f.write(str(player.policy_net.conv2) + "\n")
        try:
            f.write(str(player.policy_net.conv3) + "\n")
        except:
            pass

        f.write(str(player.policy_net.fc1) + "\n")
        f.write(str(player.policy_net.fc2) + "\n")
        f.write("Size: " + str(player.size) + "\n")
        f.write("Gamma: " + str(player.gamma) + "\n")
        f.write("Batch size: " + str(player.batch_size) + "\n")
        f.write("Optimizer: " + str(player.optimizer) + "\n")
        f.write("Ignore opponent:" + str(ignore_opponent) + "\n")

glie_a = args.glie_a
target_update = args.target_update

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

win1 = 0
wins = []
test_wins = []
avg_over = 50

RA_actions = 0
rewards = []
for i in range(start_episode, episodes):
    done = False
    if glie_a <= 0 or i % test_episodes == 0:
        eps = 0
    else:
        eps = glie_a / (glie_a + i)
        if eps < 0.1:
            if random.random() > 0.5:
                eps = 0.1
            else:
                eps = 0

    state, _ = env.reset()
    state = process_state(state, player.size, ignore_opponent)
    state_diff = 2 * state - state

    actions = 0
    while not done:
        actions += 1
        # Get the actions from both SimpleAIs
        action1 = player.get_action(state_diff, eps)
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (next_state, ob2), (rew1, rew2), done, info = env.step((action1, action2))

        next_state = process_state(next_state, player.size, ignore_opponent)
        next_state_diff = 2 * next_state - state

        if rew1 == 10:
            win1 += 1

        player.store_transition(state_diff, action1, next_state_diff, rew1, done)

        state_diff = next_state_diff
        state = next_state

        if done:
            player.update_network()
            rewards.append(rew1)

            if rew1 == 10:
                wins.append(1)
                if eps == 0:
                    test_wins.append(1)
            else:
                wins.append(0)
                if eps == 0:
                    test_wins.append(0)

            if len(wins) > avg_over:
                wins = wins[-avg_over:]

            if len(test_wins) > avg_over:
                test_wins = test_wins[-avg_over:]

            if i % target_update == 0:
                player.update_target_network()
                print("Target network updated!")

            observation = env.reset()
            RA_actions = 0.9 * RA_actions + 0.1 * actions
            print("episode {} over. TWR: {:.3f}. RAA: {:.3f}. Ep: {:.3f}".format(i, np.mean(test_wins), RA_actions, eps))

    if i % 30 == 0:
        print("Model:", model_name)

    if i % 100 == 0 or (len(test_wins) > 20 and np.mean(test_wins) > model_info["test_WR"] * 1.03 > 0.7):
        print("Model name:", model_name)
        chosen_actions = player.chosen_actions
        if np.sum(chosen_actions) != 0:
            chosen_actions /= np.sum(chosen_actions)

        chosen_actions = np.round(chosen_actions, 2)
        print("Action distribution:", list(chosen_actions))

        if args.save:
            with open("DQN_SAA/" + model_name + "/rewards.p", "wb") as f:
                pickle.dump(rewards, f)

            better_model = True
            if glie_a / (glie_a + i) <= 0.1:
                with open("DQN_SAA/" + model_name + "/model_info.p", "rb") as f:
                    prev_model_info = pickle.load(f)
                    if "test_WR" in prev_model_info:
                        if np.mean(test_wins) < prev_model_info["test_WR"] < 1.0:
                            better_model = False

            if better_model:
                test_WR = np.mean(test_wins)
                torch.save(player.policy_net, "DQN_SAA/" + model_name + "/policy_net.pth")
                print("Model", model_name, "saved!")
                print("New test win-rate: {:.2f}".format(test_WR))
                failed = 0
            else:
                test_WR = prev_model_info["test_WR"]
                failed += 1
                print("Did not save model: no improvements made! Failed", failed, "times...")
                print("Old test win-rate: {:.2f}".format(test_WR))

            if failed >= restart_after_fails:
                print("Improving model failed", restart_after_fails, "times, resetting to old version!")
                player.policy_net = torch.load("DQN_SAA/" + model_info["model_name"] + "/policy_net.pth")
                failed = 0
                test_wins = []
                continue

            model_info["model_name"] = model_name
            model_info["size"] = player.size
            model_info["episode"] = i
            model_info["fc1_size"] = player.fc1_size
            model_info["test_WR"] = test_WR
            model_info["ignore_opponent"] = ignore_opponent

            with open("DQN_SAA/" + model_name + "/model_info.p", "wb") as f:
                pickle.dump(model_info, f)

            with open("DQN_SAA/" + model_name + "/performance.txt", "a") as f:
                f.write("episode {} over. RWR: {:.3f}. RAA: {:.3f}. Ep: {:.3f} \n".format(i, np.mean(wins), RA_actions, eps))
                f.write("Test win rate {:.3f}.\n".format(test_WR))