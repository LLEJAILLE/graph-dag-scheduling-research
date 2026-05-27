import torch
import torch.nn as nn

import os
from sys import path
path.append("..")

from dotenv import load_dotenv
from pathlib import Path
import matplotlib.pyplot as plt

from MCS.multi_cpu_sim import MCS
from training.buildData import encode_state

env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)

MAX_GRID_SIZE = int(os.getenv('MAX_GRID_SIZE'))
MAX_CPUS = MAX_GRID_SIZE * MAX_GRID_SIZE
MAX_TASKS = int(os.getenv('MAX_TASKS'))

class MLPNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.net(x)
    

def modelPolicy(env, model, device):
    state = torch.tensor(
        encode_state(env.env),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(state)

    best = logits.argmax().item()

    # WAIT
    if best == MAX_TASKS * MAX_CPUS:
        return None, None

    task = best // MAX_CPUS
    cpu = best % MAX_CPUS

    if not env.is_valid_action(task, cpu):
        return None, None

    return task, cpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 127
OUTPUT_SIZE = MAX_TASKS * MAX_CPUS + 1

model = MLPNN(INPUT_SIZE, OUTPUT_SIZE)

model.load_state_dict(
    torch.load(
        "../models/cpu_scheduler_model_MLP_2.pth",
        map_location=device,
        weights_only=False
    )
)

model.to(device)
model.eval()

NUM_EPISODES = 50

times = []

for seed in range(NUM_EPISODES):

    env = MCS(seed + 1420, render_mode="machine")
    env.generate_environment()

    nb_steps = 0

    while not env.is_done:
        nb_steps += 1

        action = modelPolicy(env, model, device)

        env.step(action)

        if nb_steps > 200:
            break

    times.append(env.total_time)
    plt.ioff()
    plt.close()

    print(f"Seed {seed + 1420} - Total time: {env.total_time}")

avg_time = sum(times) / len(times)

print("\n===================================")
print(f"Average total time: {avg_time}")
print("===================================")