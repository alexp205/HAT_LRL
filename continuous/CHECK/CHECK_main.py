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

import CHECK_agents as agents
import data_handler
import gym

data_logger = data_handler.DataHandler(datafile_name)

is_demo = False
load_models = False
render = False

# tasks
task = "MountainCarContinuous-v0" #"Pendulum-v0", "BipedalWalker-v3"]

# hyperparams
# - overall
batch_size = 128
ep_lim = 300 #500
test_run_lim = 10000
log_period = 500
# - DDPG-specific
mem_len = 50000 #20000
explore_steps = 0
gamma = 0.99
critic_alpha = 0.0002
actor_alpha = 0.0001
tau_alpha = 0.0001
tau = 0.001
epsilon = 1.
epsilon_decay = 0.9995
epsilon_min = 0.01
i_mod = 10.

# NOTE for mask plotting, try to comment out when not using
#fig, ax = plt.subplots()

def main():
    if not is_demo:
        # TRAINING
        print("doing training")
        env = gym.make(task)
        env.seed(seed_val)

        s_shape = env.observation_space.shape
        if isinstance(env.action_space, gym.spaces.Discrete):
            a_shape = (env.action_space.n,)
        else:
            a_shape = env.action_space.shape

        agent = agents.PlayerAgent(gamma, critic_alpha, actor_alpha, tau_alpha, seed_val, mem_len, epsilon, epsilon_decay, epsilon_min, s_shape, a_shape, i_mod)
        if load_models:
            agent.load()

        s = env.reset()

        agent.set_up_task(s_shape, a_shape)

        temp_ep = 0
        check_ep = 0
        run = 0
        total_runs = 0
        dt_start = datetime.now()
        ep_r = 0

        while temp_ep < ep_lim:
            if render:
                env.render()

            # get action
            a = agent.act(s, False)

            # perform action and advance env
            s_prime, r, done, _ = env.step(a)
            if done and (r > 0):
                r = 100
            else:
                r = 0

            run += 1
            total_runs += 1
            ep_r += r

            #rho, rho_updated = agent.update_dens_models(s)

            #agent.store_experience(s, a, r, s_prime, done, temp_ep, run, rho, rho_updated)
            agent.store_experience(s, a, r, s_prime, done, temp_ep, run, None, None)
            s = np.copy(s_prime)

            agent.update_models(batch_size)
            agent.update_target_models(tau)

            # do data logging
            if 0 == total_runs % log_period:
                walltime = datetime.now() - dt_start
                if "CartPole-v1" == task:
                    ep_diff = temp_ep - check_ep
                    log_data = (total_runs, float(ep_r)/float(ep_diff+1), walltime.total_seconds())
                    check_ep = temp_ep
                else:
                    log_data = (total_runs, float(ep_r)/float(run), walltime.total_seconds())
                data_logger.store_train_data(log_data, 1)
                data_logger.save_train_data(task, 1)

                print("period r = {}".format(ep_r))

                run = 0
                ep_r = 0

            if done:
                temp_ep += 1

                ##r_history.appendleft(ep_r)
                ##if statistics.mean(r_history) > 9.9:
                ##if ep_r > best_r:
                ##    best_r = ep_r
                ##    #break
                #if 0 == temp_ep % 10:
                #    agent.save()

                print("(ep {})".format(temp_ep))

                s = env.reset()

        env.close()

        # TESTING
        print("doing phase testing")
        #for idx, task in enumerate(test_task_arr):
        env = gym.make(task)
        env.seed(seed_val+69)
        s_shape = env.observation_space.shape
        if isinstance(env.action_space, gym.spaces.Discrete):
            a_shape = (env.action_space.n,)
        else:
            a_shape = env.action_space.shape
        s = env.reset()

        agent.set_up_task(s_shape, a_shape)

        temp_ep = 0
        check_ep = 0
        run_lim = test_run_lim
        run = 0
        total_runs = 0
        dt_start = datetime.now()
        ep_r = 0

        while total_runs < run_lim:
            if render:
                env.render()
            a = agent.act(s, True)
            s_prime, r, done, _ = env.step(a)
            if 3 == len(s_shape):
                s_prime = img_preprocess(s_prime)
            s = np.copy(s_prime)

            run += 1
            total_runs += 1
            ep_r += r

            # do data logging
            if 0 == total_runs % log_period:
                walltime = datetime.now() - dt_start
                if "CartPole-v1" == task:
                    ep_diff = temp_ep - check_ep
                    log_data = (total_runs, float(ep_r)/float(ep_diff+1), walltime.total_seconds())
                    check_ep = temp_ep
                else:
                    log_data = (total_runs, float(ep_r)/float(run), walltime.total_seconds())
                data_logger.store_test_data(log_data, 1)
                data_logger.save_test_data(task, 1, 0)

                print("TEST period r = {}".format(ep_r))

                run = 0
                ep_r = 0

            if done or (total_runs == run_lim):
                temp_ep += 1

                print("TEST (ep {})".format(temp_ep))

                s = env.reset()

        data_logger.reset_test_data()

        env.close()
    else:
        pass

if __name__ == "__main__":
    main()
