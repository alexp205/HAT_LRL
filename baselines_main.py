import sys
seed_val = int(sys.argv[1])
datafile_name = sys.argv[2]
import os
os.environ['PYTHONHASHSEED']=str(seed_val)
import numpy as np
np.random.seed(seed_val)
import random
random.seed(seed_val)
import torch
torch.manual_seed(seed_val)

from datetime import datetime
import collections
import statistics
import matplotlib.pyplot as plt
import cv2

import baselines_agents as agents
import data_handler
import gym
import gridenv_walldeath as gridenv

data_logger = data_handler.DataHandler(datafile_name)

load_models = False
render = False

# tasks
available_tasks = ["LunarLander-v2"] # NOTE change this
#available_tasks = ["gridworld"] # NOTE change this
#available_tasks = ["CartPole-v1"] # NOTE change this
task_arr = [available_tasks[0]]
test_task_arr = [available_tasks[0]]

# hyperparams
# - overall
batch_size_dict = {available_tasks[0]: 128}
run_lim_dict = {available_tasks[0]: 200000} # NOTE change this
#run_lim_dict = {available_tasks[0]: 20000} # NOTE change this
#run_lim_dict = {available_tasks[0]: 20000} # NOTE change this
test_run_lim_dict = {available_tasks[0]: 10000} # NOTE change this
#test_run_lim_dict = {available_tasks[0]: 10000} # NOTE change this
#test_run_lim_dict = {available_tasks[0]: 10000} # NOTE change this
# - DDQN-specific
mem_len = 20000 # TODO apparently this is too much, need to cut down dramatically (~1000 for each)
explore_steps = 0
gamma = 0.99
alpha = 0.001
tau = 0.001
epsilon = 1.
epsilon_decay = 0.999
epsilon_min = 0.01
smax = 400
lamb = 0.75
thresh_emb = 6
thresh_cosh = 50
clipgrad = 10000

# NOTE for mask plotting, try to comment out when not using
#fig, ax = plt.subplots()

def img_preprocess(obs):
    # manual
    #obs = obs[35:195]
    #s = np.dot(obs[::2,::2,:3], [0.2989, 0.5870, 0.1140]) # ref: https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python

    # auto
    obs = cv2.cvtColor(cv2.resize(obs, (80, 80)), cv2.COLOR_BGR2GRAY)
    #obs = cv2.cvtColor(cv2.resize(obs, (80, 80), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    #obs = cv2.cvtColor(cv2.pyrDown(obs, dstsize=(80, 80)), cv2.COLOR_BGR2GRAY)
    #ret, obs = cv2.threshold(obs, 1, 255, cv2.THRESH_BINARY)
    # TODO use frame stacking?

    #temp_s = obs
    #print("final image:")
    #print(temp_s.shape)
    #plt.imshow(temp_s, interpolation="nearest")
    #plt.show()
    #input("wait")

    obs = np.expand_dims(obs, axis=0)

    return obs

def main():
    # TRAINING
    print("doing training")
    for task in task_arr:
        print("in task:")
        print(task)

        if "gridworld" == task:
            env = gridenv.GridEnvSim(10, 10, False, 1, False, True, 1, render)
            # TODO for final paper replace this w/ the harder (but more accepted) gym-minigrid
        else:
            env = gym.make(task)
            env.seed(seed_val)
        s_shape = env.observation_space.shape
        if isinstance(env.action_space, gym.spaces.Discrete):
            a_shape = (env.action_space.n,)
        else:
            a_shape = env.action_space.shape
        s = env.reset()
        if 3 == len(s_shape):
            s = img_preprocess(s)
        agent = agents.PlayerAgent(gamma, alpha, seed_val, mem_len, epsilon, epsilon_decay, epsilon_min, s_shape, a_shape)
        if load_models:
            agent.load()

        temp_ep = 0
        run_lim = run_lim_dict[task]
        batch_size = batch_size_dict[task]
        run = 0
        total_runs = 0
        dt_start = datetime.now()
        ep_r = 0

        while total_runs < run_lim:
            if render:
                env.render()

            # get action
            a = agent.act(s)

            # perform action and advance env
            s_prime, r, done, _ = env.step(a)
            if 3 == len(s_shape):
                s_prime = img_preprocess(s_prime)

            run += 1
            total_runs += 1
            ep_r += r

            agent.store_experience(s, a, r, s_prime, done, temp_ep, run)
            s = np.copy(s_prime)

            agent.update_target_models(tau)
            agent.update_models(batch_size, total_runs, run_lim)

            if done or (total_runs == run_lim):
                temp_ep += 1

                # do data logging
                walltime = datetime.now() - dt_start
                if "CartPole-v1" == task:
                    log_data = (temp_ep-1, float(ep_r), total_runs, walltime.total_seconds())
                else:
                    log_data = (temp_ep-1, float(ep_r)/float(run), total_runs, walltime.total_seconds())
                data_logger.store_train_data(log_data, 0)
                data_logger.save_train_data(task, 0)

                ##r_history.appendleft(ep_r)
                ##if statistics.mean(r_history) > 9.9:
                ##if ep_r > best_r:
                ##    best_r = ep_r
                ##    #break
                #if 0 == temp_ep % 10:
                #    agent.save()

                print("(ep {}) r = {}".format(temp_ep, ep_r))

                run = 0
                ep_r = 0

                s = env.reset()
                if 3 == len(s_shape):
                    s = img_preprocess(s)

        env.close()

    # TESTING
    print("doing testing")
    for idx, task in enumerate(test_task_arr):
        print("in task:")
        print(task)

        if "gridworld" == task:
            env = gridenv.GridEnvSim(10, 10, False, 1, False, True, 1, render)
            # TODO for final paper replace this w/ the harder (but more accepted) gym-minigrid
        else:
            env = gym.make(task)
            env.seed(seed_val+idx+69)
        s_shape = env.observation_space.shape
        if isinstance(env.action_space, gym.spaces.Discrete):
            a_shape = (env.action_space.n,)
        else:
            a_shape = env.action_space.shape
        s = env.reset()
        if 3 == len(s_shape):
            s = img_preprocess(s)

        temp_ep = 0
        run_lim = test_run_lim_dict[task]
        run = 0
        total_runs = 0
        dt_start = datetime.now()
        ep_r = 0

        while total_runs < run_lim:
            if render:
                env.render()
            a = agent.act(s)
            s_prime, r, done, _ = env.step(a)
            if 3 == len(s_shape):
                s_prime = img_preprocess(s_prime)
            s = np.copy(s_prime)

            run += 1
            total_runs += 1
            ep_r += r

            if done or (total_runs == run_lim):
                temp_ep += 1

                # do data logging
                walltime = datetime.now() - dt_start
                if "CartPole-v1" == task:
                    log_data = (temp_ep-1, float(ep_r), total_runs, walltime.total_seconds())
                else:
                    log_data = (temp_ep-1, float(ep_r)/float(run), total_runs, walltime.total_seconds())
                data_logger.store_test_data(log_data, 0)
                data_logger.save_test_data(task, 0)

                print("TEST (ep {}) r = {}".format(temp_ep, ep_r))

                run = 0
                ep_r = 0

                s = env.reset()
                if 3 == len(s_shape):
                    s = img_preprocess(s)

        env.close()

if __name__ == "__main__":
    main()
