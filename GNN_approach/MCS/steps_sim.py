# MCS/steps_sim.py
# This file contains the core logic for simulating one step in the MCS environment.

def is_valid_action(self, task, cpu):
        if task not in self.env['graph_tasks']:
            return False

        if cpu not in self.env['cpu_state']:
            return False

        task_data = self.env['graph_tasks'][task]

        if task_data['status'] != 'pending':
            return False

        if any(self.env['graph_tasks'][dep]['status'] != 'completed'
            for dep in task_data['dependencies']):
            return False

        if self.env['cpu_state'][cpu]['current_task'] is not None:
            return False

        return True

def step_sim(self, action):
    done = False
    self.total_time += 1

    completed_this_step = 0

    for cpu_id, state in self.env['cpu_state'].items():
        if state['current_task'] is not None:
            state['time_remaining'] -= 1
            if state['time_remaining'] <= 0:
                completed_task = state['current_task']
                self.env['graph_tasks'][completed_task]['status'] = 'completed'
                state['current_task'] = None
                completed_this_step += 1

    if action is not None:
        task, cpu = action

        if task is not None and cpu is not None and is_valid_action(self, task, cpu):
            self.env['cpu_state'][cpu]['current_task'] = task
            self.env['cpu_state'][cpu]['time_remaining'] = self.env['graph_tasks'][task]['duration']
            self.env['graph_tasks'][task]['status'] = 'in_progress'

    if all(d['status'] == 'completed' for d in self.env['graph_tasks'].values()):
        done = True
        self.is_done = True
        return None, done, {}

    obs = self.env
  
    return obs, done, {}
