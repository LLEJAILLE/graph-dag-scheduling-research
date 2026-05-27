import random
import os

from sys import path
path.append("..")
from MCS.multi_cpu_sim import MCS

from dotenv import load_dotenv
from pathlib import Path

env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)

MIN_GRID_SIZE = int(os.getenv('MIN_GRID_SIZE'))
MAX_GRID_SIZE = int(os.getenv('MAX_GRID_SIZE'))
MIN_TASKS = int(os.getenv('MIN_TASKS'))
MAX_TASKS = int(os.getenv('MAX_TASKS'))
MAX_CPUS = MAX_GRID_SIZE * MAX_GRID_SIZE
MAX_DURATION = int(os.getenv('MAX_DURATION'))
MAX_STATUS = int(os.getenv('MAX_STATUS'))
MAX_TASK_LINKS = MAX_TASKS - 1
MAX_CPU_NEIGHBORS = int(os.getenv('MAX_CPU_NEIGHBORS'))

def copy(env):
    return {
        'grid_size': env['grid_size'],
        'num_cpus': env['num_cpus'],
        'graph_cpu': {k: list(v) for k, v in env['graph_cpu'].items()},
        'cpu_state': {k: v.copy() for k, v in env['cpu_state'].items()},
        'graph_tasks': {k: v.copy() for k, v in env['graph_tasks'].items()},
    }

def heuristicPolicy(env):
    ready_tasks = env.get_ready_tasks()
    if not ready_tasks:
        return None, None

    available_cpus = [
        cpu_id for cpu_id, state in env.env['cpu_state'].items()
        if state['current_task'] is None
    ]
    if not available_cpus:
        return None, None

    def score(t):
        task = env.env['graph_tasks'][t]
        return (
            len(task['dependents']) 
            + task['duration'] * 0.5
            + random.random() * 0.01
        )
    
    best_task = max(ready_tasks, key=score)

    def cpu_score(cpu_id):
        neighbors = env.env['graph_cpu'][cpu_id]
        busy_neighbors = sum(
            1 for n in neighbors
            if env.env['cpu_state'][n]['current_task'] is not None
        )
        return -busy_neighbors + random.random() * 0.01

    cpu_id = max(available_cpus, key=cpu_score)

    return best_task, cpu_id

def encode_state(env):
    tasks = env['graph_tasks']
    cpus = env['cpu_state']
    graph_cpu = env['graph_cpu']

    task_features = []

    for i in range(MAX_TASKS):
        if i in tasks:
            t = tasks[i]
            status = 0 if t['status'] == 'pending' else (1 if t['status'] == 'in_progress' else 2)

            task_features.extend([
                min(t['duration'] / MAX_DURATION, 1.0),
                status / MAX_STATUS,
                min(len(t['dependencies']) / MAX_TASK_LINKS, 1.0),
                min(len(t['dependents']) / MAX_TASK_LINKS, 1.0)
            ])
        else:
            task_features.extend([0, 0, 0, 0])  # padding

    cpu_features = []

    for i in range(MAX_CPUS):
        if i in cpus:
            c = cpus[i]
            is_free = 1 if c['current_task'] is None else 0

            cpu_features.extend([
                is_free,
                min(c['time_remaining'] / MAX_DURATION, 1.0),
                min(len(graph_cpu[i]) / MAX_CPU_NEIGHBORS, 1.0)
            ])
        else:
            cpu_features.extend([0, 0, 0])  # padding

    return task_features + cpu_features


def encode_action(action):
    if action is None or action == (None, None):
        return MAX_TASKS * MAX_CPUS  # WAIT

    t, c = action
    return t * MAX_CPUS + c
