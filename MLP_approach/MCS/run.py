import random
from MCS.multi_cpu_sim import MCS
import matplotlib.pyplot as plt

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


def randomPolicyFull(env):
    task = random.randint(0, env.num_tasks - 1)
    cpu = random.randint(0, env.num_cpus - 1)
    return task, cpu


NUM_EPISODES = 50

times = []

for seed in range(NUM_EPISODES):

    env = MCS(seed + 1420, render_mode="human")
    env.generate_environment()

    nb_steps = 0

    while not env.is_done:
        nb_steps += 1

        action = heuristicPolicy(env)

        env.step(action)

        if nb_steps > 1000:
            print(f"Seed {seed + 1420} - Episode stopped after 1000 steps (possible infinite loop).")
            break

    times.append(env.total_time)
    plt.ioff()
    plt.close()
    print(f"Seed {seed + 1420} - Total time: {env.total_time}")


avg_time = sum(times) / len(times)

print("\n===================================")
print(f"Average total time: {avg_time}")
print("===================================")