import pickle as p
import os

DIRs = [x for x in os.listdir("DQN_SAA/") if os.path.isdir("DQN_SAA/" + x) and "cache" not in x]
i = 0
for DIR in DIRs:
    print(i, DIR)
    i += 1

model_ind = int(input("Model number:"))
model_name = DIRs[model_ind]

with open("DQN_SAA/" + model_name + "/model_info.p", "rb") as f:
    model_info = p.load(f)

print("Current WR: {:.3f}".format(model_info["test_WR"]))
new_WR = float(input("New winrate:"))

model_info["test_WR"] = new_WR

with open("DQN_SAA/" + model_name + "/model_info.p", "wb") as f:
    p.dump(model_info, f)
