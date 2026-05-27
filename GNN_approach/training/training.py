import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

import os
import sys
import time

from sys import path
path.append("..")
from MCS.multi_cpu_sim import MCS
from buildData import encode_action, heuristicPolicy

from dotenv import load_dotenv
from pathlib import Path

env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)

MIN_GRID_SIZE = int(os.getenv('MIN_GRID_SIZE'))
MAX_GRID_SIZE = int(os.getenv('MAX_GRID_SIZE'))
MAX_CPUS = MAX_GRID_SIZE * MAX_GRID_SIZE
MAX_TASKS = int(os.getenv('MAX_TASKS'))
MAX_DURATION = 10
MAX_TASK_LINKS = MAX_TASKS - 1

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")



NUM_SEEDS = 1000
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_CHANNELS = 16
IN_CHANNELS = 5
MODEL_NAME = "GCN_BC_v3_d3.pth"


graph_dataset = []

class TaskSchedulingGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_actions):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels + MAX_CPUS * 2, num_actions)

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index

        # Message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph-level embedding
        x = global_mean_pool(x, data.batch)

        cpu_features = data.cpu_features.view(x.size(0), -1)

        x = torch.cat([x, cpu_features], dim=1)

        # Action prediction
        x = self.lin(x)

        return x


def build_graph(env, action):
    tasks = env.env['graph_tasks']

    x = []
    edge_index = [[], []]

    cpu_features = []

    for cpu_id, cpu in env.env['cpu_state'].items():
        is_free = 1 if cpu['current_task'] is None else 0

        cpu_features.extend([
            is_free,
            cpu['time_remaining'] / MAX_DURATION
        ])

    while len(cpu_features) < MAX_CPUS * 2:
        cpu_features.extend([0, 0])

    cpu_features = torch.tensor(
        cpu_features,
        dtype=torch.float
    )


    for task_id, task in tasks.items():

        status = (
            0 if task['status'] == 'pending'
            else 1 if task['status'] == 'in_progress'
            else 2
        )

        is_ready = 1 if env.is_task_ready(task_id) else 0

        features = [
            task['duration'] / MAX_DURATION,
            status / 2,
            len(task['dependencies']) / MAX_TASK_LINKS,
            len(task['dependents']) / MAX_TASK_LINKS,
            is_ready
        ]

        x.append(features)

        for dep in task['dependencies']:
            edge_index[0].append(dep)
            edge_index[1].append(task_id)


    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    y = torch.tensor([encode_action(action)], dtype=torch.long)

    graph = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        cpu_features=cpu_features
    )

    return graph

def build_dataset():
    for seed in range(NUM_SEEDS):
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
                action = heuristicPolicy(env)
                graph = build_graph(env, action)

                graph_dataset.append(graph)

                env.step(action)
                

if __name__ == "__main__":
    start = time.time()

    build_dataset()
    print("=================================================")
    print(f"Generated {len(graph_dataset)} graph samples.")
    print("=================================================")

    GNN = TaskSchedulingGNN(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        num_actions=MAX_TASKS * MAX_CPUS + 1
    ).to(device)

    optimizer = torch.optim.Adam(GNN.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = NUM_EPOCHS

    print("Starting training...")
    print("=================================================")

    loader = DataLoader(
        graph_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    log = []

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = GNN(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(graph_dataset)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        log.append(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(GNN.state_dict(), f"{MODEL_NAME}_in_training.pth")

    
    end = time.time()
    print("=================================================")
    print(f"Training completed in {(end - start) / 60:.2f} minutes.")
    print("=================================================")

    log.append(f"Training completed in {(end - start) / 60:.2f} minutes.")
    with open(f"{MODEL_NAME}_epochs_loss_log.txt", "w", encoding="utf-16") as f:
        for line in log:
            f.write(line + "\n")

    torch.save(GNN.state_dict(), MODEL_NAME)
