import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool

import os
from sys import path
path.append("..")

from dotenv import load_dotenv
from pathlib import Path
import matplotlib.pyplot as plt

from MCS.multi_cpu_sim import MCS

env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)

MAX_GRID_SIZE = int(os.getenv('MAX_GRID_SIZE'))
MAX_CPUS = MAX_GRID_SIZE * MAX_GRID_SIZE
MAX_TASKS = int(os.getenv('MAX_TASKS'))

MAX_DURATION = 10
MAX_TASK_LINKS = MAX_TASKS - 1

HIDDEN_CHANNELS = 16
IN_CHANNELS = 5

class TaskSchedulingGNN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, num_actions):
        super().__init__()

        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)

        self.lin = Linear(
            hidden_channels + MAX_CPUS * 2,
            num_actions
        )

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index

        # GAT message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph embedding
        x = global_mean_pool(x, data.batch)

        # CPU features
        cpu_features = data.cpu_features.view(
            -1,
            MAX_CPUS * 2
        )

        # Merge graph + CPU state
        x = torch.cat([x, cpu_features], dim=1)

        # Action logits
        x = self.lin(x)

        return x


def build_graph(env):

    tasks = env.env['graph_tasks']

    x = []
    edge_index = [[], []]

    cpu_features = []

    # CPU FEATURES
    for cpu_id, cpu in env.env['cpu_state'].items():

        is_free = (
            1 if cpu['current_task'] is None
            else 0
        )

        cpu_features.extend([
            is_free,
            cpu['time_remaining'] / MAX_DURATION
        ])

    # Padding CPU features
    while len(cpu_features) < MAX_CPUS * 2:
        cpu_features.extend([0, 0])

    cpu_features = torch.tensor(
        cpu_features,
        dtype=torch.float
    )

    # TASK GRAPH
    for task_id, task in tasks.items():

        status = (
            0 if task['status'] == 'pending'
            else 1 if task['status'] == 'in_progress'
            else 2
        )

        is_ready = (
            1 if env.is_task_ready(task_id)
            else 0
        )

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

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    )

    graph = Data(
        x=x,
        edge_index=edge_index,
        cpu_features=cpu_features
    )

    return graph


def modelPolicy(env, model, device):

    graph = build_graph(env)

    # Single graph batch
    graph.batch = torch.zeros(
        graph.x.size(0),
        dtype=torch.long
    )

    graph = graph.to(device)

    with torch.no_grad():
        logits = model(graph)

    logits = logits.squeeze(0)

    # VALID ACTION MASKING
    valid_actions = []

    for task in range(MAX_TASKS):
        for cpu in range(MAX_CPUS):
            if env.is_valid_action(task, cpu):
                action_id = (task * MAX_CPUS + cpu)
                valid_actions.append(action_id)

    # No valid action => WAIT
    if not valid_actions:
        return None, None

    # Select best VALID action only
    best_action = max(
        valid_actions,
        key=lambda a: logits[a].item()
    )

    task = best_action // MAX_CPUS
    cpu = best_action % MAX_CPUS

    return task, cpu


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

model = TaskSchedulingGNN(
    in_channels=IN_CHANNELS,
    hidden_channels=HIDDEN_CHANNELS,
    num_actions=MAX_TASKS * MAX_CPUS + 1
)

model.load_state_dict(
    torch.load(
        "../models/GAT_BC_v1/GAT_BC_v2_d3_5000dt.pth_in_training.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

NUM_EPISODES = 20

times = []

for seed in range(NUM_EPISODES):

    env = MCS(seed + 5000, render_mode="machine")
    env.generate_environment()

    nb_steps = 0

    while not env.is_done:

        nb_steps += 1

        action = modelPolicy(
            env,
            model,
            device
        )

        env.step(action)

        # Infinite loop protection
        if nb_steps > 1000:
            break

    times.append(env.total_time)

    plt.ioff()
    plt.close()

    print(
        f"Seed {seed + 5000} "
        f"- Total time: {env.total_time}"
    )

avg_time = sum(times) / len(times)

print("\n===================================")
print(f"Average total time: {avg_time}")
print("===================================")