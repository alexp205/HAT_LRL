import numpy as np
import pandas as pd

class DataHandler:
    def __init__(self, fname):
        self.f_dir = "./data/"
        self.f_name = fname

        # info for train
        self.episode_data = {}
        self.r_data = {}
        self.num_runs = {}
        self.num_time = {}

        # info for test
        self.test_episode_data = {}
        self.test_r_data = {}
        self.test_num_runs = {}
        self.test_num_time = {}

    def store_train_data(self, data, task_id):
        episode_val, r_val, run_val, time_val = data

        if task_id not in self.episode_data:
            self.episode_data[task_id] = [episode_val]
            self.r_data[task_id] = [r_val]
            self.num_runs[task_id] = [run_val]
            self.num_time[task_id] = [time_val]
        else:
            self.episode_data[task_id].append(episode_val)
            self.r_data[task_id].append(r_val)
            self.num_runs[task_id].append(run_val)
            self.num_time[task_id].append(time_val)

    def save_train_data(self, task, task_id):
        data_dict = {'episode': self.episode_data[task_id], 'r': self.r_data[task_id], 'runs': self.num_runs[task_id], 'time': self.num_time[task_id]}

        df = pd.DataFrame(data_dict)

        f = self.f_dir + self.f_name + "_" + task + ".csv"

        df.to_csv(f, header=False, index=False)

    def store_test_data(self, data, task_id):
        episode_val, r_val, run_val, time_val = data

        if task_id not in self.test_episode_data:
            self.test_episode_data[task_id] = [episode_val]
            self.test_r_data[task_id] = [r_val]
            self.test_num_runs[task_id] = [run_val]
            self.test_num_time[task_id] = [time_val]
        else:
            self.test_episode_data[task_id].append(episode_val)
            self.test_r_data[task_id].append(r_val)
            self.test_num_runs[task_id].append(run_val)
            self.test_num_time[task_id].append(time_val)

    def save_test_data(self, task, task_id):
        data_dict = {'episode': self.test_episode_data[task_id], 'r': self.test_r_data[task_id], 'runs': self.test_num_runs[task_id], 'time': self.test_num_time[task_id]}

        df = pd.DataFrame(data_dict)

        f = self.f_dir + "TEST_" + self.f_name + "_" + task + ".csv"

        df.to_csv(f, header=False, index=False)
