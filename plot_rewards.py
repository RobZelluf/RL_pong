import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

DIRs = [x for x in os.listdir("DQN_SAA/") if os.path.isdir("DQN_SAA/" + x) and "cache" not in x]
i = 0
for DIR in DIRs:
    print(i, DIR)
    i += 1

model_ind = int(input("Model number:"))
model_name = DIRs[model_ind]

with open("DQN_SAA/" + model_name + "/rewards.p", "rb") as f:
    rewards = pickle.load(f)

cumulative_rewards = []
for i in range(len(rewards)):
    cumulative_rewards.append(sum(rewards[:i]))

average_rewards = []
for i in range(len(rewards)):
    average_rewards.append(np.mean(rewards[:i]))

plt.figure()
plt.plot(cumulative_rewards)
plt.xlabel("Number of episodes")
plt.ylabel("Cumulative reward")
plt.figure()
plt.plot(average_rewards)
plt.xlabel("Number of episodes")
plt.ylabel("Average reward per episode")
plt.show()
