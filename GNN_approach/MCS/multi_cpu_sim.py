import random
import networkx as nx
import matplotlib.pyplot as plt

# imports
from .print_graphs import render_human
from .steps_sim import step_sim
from .generate_env import generate_mcs_environment, is_valid_a

class MCS:
    # define MCS variables
    def __init__(self, seed, render_mode="machine"):
        self.seed = seed
        self.rng = random.Random(seed)
        self.env = {}
        self.is_done = False
        self.total_time = 0
        self.pos = None
        self.G = None

        self.render_mode = render_mode

        if self.render_mode == "human":
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(14, 8))

    def build_hierarchical_pos(self, graph_tasks):
        levels = {}
        def get_level(task):
            if task in levels:
                return levels[task]

            deps = graph_tasks[task]["dependencies"]
            if not deps:
                levels[task] = 0
            else:
                levels[task] = 1 + max(get_level(d) for d in deps)

            return levels[task]

        for t in graph_tasks:
            get_level(t)

        level_nodes = {}
        for node, level in levels.items():
            level_nodes.setdefault(level, []).append(node)

        pos = {}
        for level, nodes in level_nodes.items():
            width = len(nodes)
            for i, node in enumerate(nodes):
                x = level
                y = -(i - width / 2)
                pos[node] = (x, y)
        return pos

    # generate environment using parameters defined in generate_env.py
    def generate_environment(self):
        generate_mcs_environment(self)
        self.pos = self.build_hierarchical_pos(self.env['graph_tasks'])
        self.G = nx.DiGraph()

        for task_id, data in self.env['graph_tasks'].items():
            self.G.add_node(task_id, duration=data["duration"], status=data["status"])

        for task_id, data in self.env['graph_tasks'].items():
            for dep in data["dependencies"]:
                self.G.add_edge(dep, task_id)


    # loop through the environment until done, using defined heuristic policy to select actions
    def step(self, action):
        step_sim(self, action)

        if self.render_mode == "human":
            if self.total_time % 2 == 0:
                render_human(self)
    
    # return list of ready tasks (pending with all dependencies completed)
    def get_ready_tasks(self):
        tasks = self.env['graph_tasks']
        return [
            t for t, d in tasks.items()
            if d['status'] == 'pending' and
            all(tasks[dep]['status'] == 'completed' for dep in d['dependencies'])
        ]
    
    # return True if task is ready to be scheduled (pending with all dependencies completed), else False
    def is_task_ready(self, task_id):
        task = self.env['graph_tasks'][task_id]
        return (
            task['status'] == 'pending' and
            all(self.env['graph_tasks'][dep]['status'] == 'completed' for dep in task['dependencies'])
        )

    # visualization of graphs and environment state
    def print_environment(self):
        print("\n--- STATE ---")
        for cpu_id, state in self.env['cpu_state'].items():
            print(f"CPU {cpu_id}: {state}")

        for task, details in self.env['graph_tasks'].items():
            print(f"Task {task}: {details}")
        
        print("-------------\n")


    def is_valid_action(self, task, cpu):
        return is_valid_a(self, task, cpu)