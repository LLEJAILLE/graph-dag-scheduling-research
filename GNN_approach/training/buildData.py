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
            + random.random() * 0.05
        )
    
    best_task = max(ready_tasks, key=score)

    def cpu_score(cpu_id):
        neighbors = env.env['graph_cpu'][cpu_id]
        busy_neighbors = sum(
            1 for n in neighbors
            if env.env['cpu_state'][n]['current_task'] is not None
        )
        return -busy_neighbors + random.random() * 0.05

    cpu_id = max(available_cpus, key=cpu_score)

    return best_task, cpu_id

def encode_action(action):
    if action is None or action == (None, None):
        return MAX_TASKS * MAX_CPUS  # WAIT

    t, c = action
    return t * MAX_CPUS + c
