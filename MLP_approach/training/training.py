import torch
import torch.nn as nn

import os
from sys import path
path.append("..")
from MCS.multi_cpu_sim import MCS
from buildData import encode_state, encode_action, copy, heuristicPolicy

from dotenv import load_dotenv
from pathlib import Path

env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)

MIN_GRID_SIZE = int(os.getenv('MIN_GRID_SIZE'))
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


def modelPolicy(env, model):
    state = torch.tensor(
        encode_state(env.env),
        dtype=torch.float32
    ).unsqueeze(0)

    logits = model(state)

    best = logits.argmax().item()

    if best == MAX_TASKS * MAX_CPUS:
        return None, None

    task = best // MAX_CPUS
    cpu = best % MAX_CPUS

    return task, cpu

encoded_dataset = []

def build_dataset():
    for seed in range(1000):
        env = MCS(seed, render_mode="machine")
        env.generate_environment()

        while not env.is_done:
            ready = env.get_ready_tasks()
            available = [
                cpu for cpu, s, in env.env['cpu_state'].items()
                if s['current_task'] is None
            ]

            if not ready or not available:
                env.step(None)
                continue
            else:
                state = copy(env.env)
                action = heuristicPolicy(env)

                x = encode_state(state)
                y = encode_action(action)

                encoded_dataset.append((x, y))

                env.step(action)

    print(f"Dataset built with {len(encoded_dataset)} samples.\n")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=================================================")
    build_dataset()

    print("Creating tensors...")
    
    X = torch.tensor(
        [x for x, _ in encoded_dataset],
        dtype=torch.float32
    )

    Y = torch.tensor(
        [y for _, y in encoded_dataset],
        dtype=torch.long
    )
    
    X = X.to(device)
    Y = Y.to(device)

    print("Initializing model...")

    model = MLPNN(
        input_size=len(X[0]),
        output_size=MAX_TASKS * MAX_CPUS + 1
    )

    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 0

    print("=================================================\n")

    print(f"Starting training ...\n")

    while criterion(model(X), Y).item() > 0.05:
        epochs += 1
        logits = model(X)

        loss = criterion(logits, Y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print(f"Epoch {epochs}, Loss: {loss.item()}")

        if epochs % 1000 == 0:
            torch.save(model.state_dict(), "cpu_scheduler_model_MLP_3_1000sce.pth")

    
    # Sauvegarde du modèle
    torch.save(model.state_dict(), "cpu_scheduler_model_MLP_3_1000sce.pth")
    print(f"Model saved after {epochs} epochs with final loss {loss.item()}.")