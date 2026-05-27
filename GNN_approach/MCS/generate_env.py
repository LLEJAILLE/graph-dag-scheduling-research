import os
from dotenv import load_dotenv
from pathlib import Path

script_dir = Path(__file__).parent
env_path = script_dir.parent / '.env'
load_dotenv(dotenv_path=env_path)

MIN_GRID_SIZE = int(os.getenv('MIN_GRID_SIZE'))
MAX_GRID_SIZE = int(os.getenv('MAX_GRID_SIZE'))
MIN_TASKS = int(os.getenv('MIN_TASKS'))
MAX_TASKS = int(os.getenv('MAX_TASKS'))

# generate environment for multi CPU scheduling problem looking like:
# CPU 0: {'current_task': None, 'time_remaining': 0}
# CPU 1: {'current_task': None, 'time_remaining': 0}
# CPU 2: {'current_task': None, 'time_remaining': 0}
# CPU 3: {'current_task': None, 'time_remaining': 0}
# Task 0: {'duration': 2, 'dependencies': [], 'dependents': [1, 5, 6, 8], 'status': 'pending'}
# Task 1: {'duration': 3, 'dependencies': [0], 'dependents': [5, 6, 7, 8], 'status': 'pending'}
# Task 2: {'duration': 10, 'dependencies': [], 'dependents': [5, 6, 8], 'status': 'pending'}
# Task 3: {'duration': 10, 'dependencies': [], 'dependents': [4, 5, 6], 'status': 'pending'}
# Task 4: {'duration': 10, 'dependencies': [3], 'dependents': [5, 6, 7], 'status': 'pending'}
# Task 5: {'duration': 7, 'dependencies': [4, 2, 3, 1, 0], 'dependents': [6, 9], 'status': 'pending'}
# Task 6: {'duration': 1, 'dependencies': [0, 2, 3, 1, 4, 5], 'dependents': [], 'status': 'pending'}
# Task 7: {'duration': 9, 'dependencies': [4, 1], 'dependents': [], 'status': 'pending'}
# Task 8: {'duration': 4, 'dependencies': [0, 1, 2], 'dependents': [9], 'status': 'pending'}
# Task 9: {'duration': 3, 'dependencies': [8, 5], 'dependents': [], 'status': 'pending'}

def generate_mcs_environment(self):
    self.grid_size = self.rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    self.num_cpus = self.grid_size * self.grid_size
    self.num_tasks = self.rng.randint(MIN_TASKS, MAX_TASKS)

    # CPU graph
    graph_cpu = {}
    for i in range(self.num_cpus):
        graph_cpu[i] = []
        if i % self.grid_size != 0:
            graph_cpu[i].append(i - 1)
        if (i + 1) % self.grid_size != 0:
            graph_cpu[i].append(i + 1)
        if i >= self.grid_size:
            graph_cpu[i].append(i - self.grid_size)
        if i < self.num_cpus - self.grid_size:
            graph_cpu[i].append(i + self.grid_size)

    # Task graph (DAG)
    graph_tasks = {}
    for i in range(self.num_tasks):
        graph_tasks[i] = {
            "duration": self.rng.randint(1, 10),
            "dependencies": [],
            "dependents": [],
            "status": "pending"
        }
        possible_parents = list(range(i))
        max_parents = min(8, len(possible_parents))
        
        if max_parents == 0:
            k = 0
        else:
            k = self.rng.randint(
                max(1, max_parents // 2),
                max_parents
            )

        parents = self.rng.sample(possible_parents, k)
        graph_tasks[i]["dependencies"] = parents
        for parent in parents:
            graph_tasks[parent]["dependents"].append(i)

    cpu_state = {
        cpu_id: {
            "current_task": None,
            "time_remaining": 0
        } for cpu_id in range(self.num_cpus)
    }

    self.env = {
        'grid_size': self.grid_size,
        'num_cpus': self.num_cpus,
        'graph_cpu': graph_cpu,
        'cpu_state': cpu_state,
        'graph_tasks': graph_tasks,
    }


def is_valid_a(self, task, cpu):

    if task is None or cpu is None:
        return False

    # task existe ?
    if task not in self.env['graph_tasks']:
        return False

    # cpu existe ?
    if cpu not in self.env['cpu_state']:
        return False

    # check if task is ready
    task_data = self.env['graph_tasks'][task]

    if task_data['status'] != 'pending':
        return False

    if any(
        self.env['graph_tasks'][dep]['status'] != 'completed'
        for dep in task_data['dependencies']
    ):
        return False

    cpu_data = self.env['cpu_state'][cpu]

    if cpu_data['current_task'] is not None:
        return False

    return True