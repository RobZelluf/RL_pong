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

plt.plot(TWR)
plt.show()

