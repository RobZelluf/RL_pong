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

RWR = []
TWR = []
RAA = []

with open("DQN_SAA/" + model_name + "/performance.txt", "r") as f:
    for line in f.readlines():
        line = line.split()
        if len(line) == 4:
            twr = line[-1][:-1]
            TWR.append(float(twr))
        else:
            rwr = line[4][:-1]
            raa = line[6][:-1]
            RWR.append(float(rwr))
            RAA.append(float(raa))

plt.figure()
x = list(range(len(RWR)))
x = [i * 100 for i in x]
RWR_a = [0]
for i in range(len(RWR)):
    RWR_a.append(0.9 * RWR_a[-1] + 0.1 * RWR[i])

RWR_a = RWR_a[1:]

plt.xlabel("Number of episodes")
plt.ylabel("Average win-rate")
plt.plot(x, RWR)
plt.plot(x, RWR_a)

plt.figure()
x = list(range(len(RAA)))
x = [i * 100 for i in x]

RAA_a = [0]
for i in range(len(RAA)):
    RAA_a.append(0.9 * RAA_a[-1] + 0.1 * RAA[i])

RAA_a = RAA_a[1:]

plt.plot(x, RAA)
plt.plot(x, RAA_a)
plt.xlabel("Number of episodes")
plt.ylabel("Average episode length (in steps)")
plt.show()

