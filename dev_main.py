# ref: https://arxiv.org/pdf/1801.01423.pdf

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

import dev_agents as agents
import data_handler
import gym
import gridenv_walldeath as gridenv

from torch.autograd import Variable

data_logger = data_handler.DataHandler(datafile_name)

is_demo = False
load_models = False
render = False

# tasks
available_tasks = ["LunarLander-v2", "gridworld", "CartPole-v1"]
#available_tasks = ["gridworld", "CartPole-v1", "LunarLander-v2"]
task_arr = [available_tasks[0], available_tasks[1], available_tasks[2]]
test_task_arr = [available_tasks[0], available_tasks[1], available_tasks[2]]
seen_task_dict = {}

# hyperparams
# - overall
batch_size_dict = {available_tasks[0]: 128, available_tasks[1]: 128, available_tasks[2]: 128}
run_lim_dict = {available_tasks[0]: 200000, available_tasks[1]: 15000, available_tasks[2]: 20000}
#run_lim_dict = {available_tasks[0]: 50, available_tasks[1]: 50, available_tasks[2]: 50}
test_run_lim_dict = {available_tasks[0]: 10000, available_tasks[1]: 10000, available_tasks[2]: 10000}
#test_run_lim_dict = {available_tasks[0]: 30, available_tasks[1]: 30, available_tasks[2]: 30}
demo_run_lim_dict = {available_tasks[0]: 10000, available_tasks[1]: 1000, available_tasks[2]: 1000}
log_period = 500
# - DDQN-specific
mem_len_dict = {0: 20000, 1: 20000, 2: 20000} # TODO apparently this is too much, need to cut down dramatically (~1000 for each)
explore_steps = 0
gamma = 0.95 #0.99
alpha = 0.001
epsilon = 1.
epsilon_decay = 0.9995 #0.999
epsilon_min = 0.01
smax = 400
lamb = 0.75
thresh_emb = 6
thresh_cosh = 50
clipgrad = 10000
surv_len_dict = {0: 5000, 1: 5000, 2: 5000}
#surv_len_dict = {0: 10, 1: 10, 2: 10}
reg_mod = 100.
reg_alpha = 5e-4 #1e-5
reg_temperature_decay = 0.99

# NOTE for mask plotting, try to comment out when not using
#fig, ax = plt.subplots()

def render_vals(env, agent, img, fig, ax, axbkgd):
    vis_s = []
    temp_s = env.get_vis_states()
    for t_s in temp_s:
        vis_s.append(np.asarray(t_s).flatten())
    vis_s = Variable(torch.from_numpy(np.asarray(vis_s)).float()).cuda()
    agent.model.eval()
    vis_q, _ = agent.model.forward(vis_s, agent.s_factor)
    vis_q = vis_q.max(dim=1)
    vis_a = vis_q[1].data.cpu().numpy()
    vis_q = vis_q[0].data.cpu().numpy()
    xv, yv, xd, yd = ([] for i in range(4))
    for idx, v_a in enumerate(vis_a):
        xv.append(idx // 10)
        yv.append(idx % 10)
        if 0 == v_a: # right
            xd.append(-0.1)
            yd.append(0)
        elif 1 == v_a: # down
            xd.append(0)
            yd.append(-0.1)
        elif 2 == v_a: # left
            xd.append(0.1)
            yd.append(0)
        else: # up
            xd.append(0)
            yd.append(0.1)

    grid = np.transpose(np.reshape(vis_q, (10,10)))
    #vectorized_arrow_drawing(xv, yv, xd, yd, 1.0)#0.15)
    # - non-blocking (hopefully) raw matplotlib
    if 0 == agent.counter % 10:
        ax.clear()
    img = ax.imshow(grid, cmap='seismic', vmin=-10, vmax=10)
    img.set_data(grid)
    temp = ax.quiver(xv, yv, xd, yd, scale=2.)#, width=0.2)
    fig.canvas.restore_region(axbkgd)
    ax.draw_artist(img)
    ax.draw_artist(temp)
    fig.canvas.blit(ax.bbox) # ref: https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot
    fig.canvas.flush_events()

    return img


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
    curr_task_id = 0

    agent = agents.PlayerAgent(gamma, alpha, seed_val, mem_len_dict, epsilon, epsilon_decay, epsilon_min, smax, lamb, thresh_emb, thresh_cosh, clipgrad, surv_len_dict, reg_mod, reg_alpha, reg_temperature_decay)
    if load_models:
        agent.load()

    if not is_demo:
        # TRAINING
        for idx, task in enumerate(task_arr):
            # TODO eventually, this should be more per-task (i.e. a tau that is tracked for each task)
            #tau = 0.001
            tau = 0.25
            tau_decay = 0.995
            tau_min = 0.001
            print("doing training")
            if task not in seen_task_dict:
                seen_task_dict[task] = curr_task_id
                curr_task_id += 1
            task_id = seen_task_dict[task]
            print("in task:")
            print(task)
            print(task_id)
            #input("wait")

            if "gridworld" == task:
                env = gridenv.GridEnvSim(10, 10, False, 1, False, True, 1, render)
                # TODO for final paper replace this w/ the harder (but more accepted) gym-minigrid

                if render:
                    input("--- WAIT (BEFORE VIS) ---")

                    grid = np.zeros((10, 10))
                    fig = plt.figure()
                    ax = fig.add_subplot(1,1,1)
                    img = ax.imshow(grid, cmap='seismic', vmin=-10, vmax=10)
                    fig.canvas.draw()
                    axbkgd = fig.canvas.copy_from_bbox(ax.bbox)
                    plt.show(block=False)
            else:
                env = gym.make(task)
                env.seed(seed_val+idx)

            s_shape = env.observation_space.shape
            if isinstance(env.action_space, gym.spaces.Discrete):
                a_shape = (env.action_space.n,)
            else:
                a_shape = env.action_space.shape

            #print("s_shape")
            #print(s_shape)
            #print("a_shape")
            #print(a_shape)
            #input("wait")

            s = env.reset()
            if 3 == len(s_shape):
                s = img_preprocess(s)

            agent.set_up_task(task_id, s_shape, a_shape)

            temp_ep = 0
            check_ep = 0
            #ep_lim = episode_lim_dict[task]
            run_lim = run_lim_dict[task]
            batch_size = batch_size_dict[task]
            run = 0
            total_runs = 0
            dt_start = datetime.now()
            ep_r = 0
            temp_ep_r = 0

            #while temp_ep < ep_lim:
            while total_runs < run_lim:
                if render:
                    if "gridworld" == task:
                        env.render()
                        img = render_vals(env, agent, img, fig, ax, axbkgd)

                # get action
                a = agent.act(s, False)

                # perform action and advance env
                s_prime, r, done, _ = env.step(a)
                if 3 == len(s_shape):
                    s_prime = img_preprocess(s_prime)

                run += 1
                total_runs += 1
                ep_r += r
                temp_ep_r += r

                agent.store_experience(s, a, r, s_prime, done, temp_ep, run, ep_r)
                s = np.copy(s_prime)

                agent.update_target_models(tau)
                agent.update_models(batch_size, total_runs, run_lim)

                tau *= tau_decay
                tau = max(tau_min, tau)

                # do data logging
                if 0 == total_runs % log_period:
                    walltime = datetime.now() - dt_start
                    if "CartPole-v1" == task:
                        ep_diff = temp_ep - check_ep
                        log_data = (total_runs, float(ep_r)/float(ep_diff+1), walltime.total_seconds())
                        check_ep = temp_ep
                    else:
                        log_data = (total_runs, float(ep_r)/float(run), walltime.total_seconds())
                    data_logger.store_train_data(log_data, task_id)
                    data_logger.save_train_data(task, task_id)

                    print("period r = {}".format(ep_r))

                    run = 0
                    ep_r = 0

                if done or (total_runs == run_lim):
                    temp_ep += 1

                    #if 0 == temp_ep % 100:
                    #    masks = agent.model.mask(task_id, agent.s_factor)
                    #    #print(masks)
                    #    for layer_idx, m in enumerate(masks):
                    #        temp_m = m.data.cpu().numpy()
                    #        temp_m = temp_m.reshape((8, 16))
                    #        #temp_m = temp_m.reshape((16, 32))
                    #        yeah = ax.imshow(temp_m, cmap='binary_r', interpolation='nearest')
                    #        #plt.show()
                    #        fig.savefig(os.path.join("./data/", task + "_mask_" + str(layer_idx) + "_" + str(temp_ep) + ".png"))
                    #        ax.clear()

                    ##r_history.appendleft(ep_r)
                    ##if statistics.mean(r_history) > 9.9:
                    ##if ep_r > best_r:
                    ##    best_r = ep_r
                    ##    #break
                    #if 0 == temp_ep % 10:
                    #    agent.save()

                    agent.store_perf(temp_ep_r)

                    #agent.update_reg_scaler()

                    print("(ep {})".format(temp_ep))

                    temp_ep_r = 0

                    s = env.reset()
                    if 3 == len(s_shape):
                        s = img_preprocess(s)

            agent.get_masks()
            agent.is_first_task = False
            env.close()

            if render:
                plt.close('all')

            # TESTING
            print("doing phase {0} testing".format(idx))
            print("envs: {0}".format(list(seen_task_dict.keys())))
            #for idx, task in enumerate(test_task_arr):
            for jdx, test_task in enumerate(seen_task_dict):
                print("in test task:")
                print(test_task)
                test_task_id = seen_task_dict[test_task]

                if "gridworld" == test_task:
                    env = gridenv.GridEnvSim(10, 10, False, 1, False, True, 1, render)
                    # TODO for final paper replace this w/ the harder (but more accepted) gym-minigrid

                    if render:
                        input("--- WAIT (BEFORE VIS) TESTING ---")

                        grid = np.zeros((10, 10))
                        fig = plt.figure()
                        ax = fig.add_subplot(1,1,1)
                        img = ax.imshow(grid, cmap='seismic', vmin=-10, vmax=10)
                        fig.canvas.draw()
                        axbkgd = fig.canvas.copy_from_bbox(ax.bbox)
                        plt.show(block=False)
                else:
                    env = gym.make(test_task)
                    env.seed(seed_val+jdx+69)
                s_shape = env.observation_space.shape
                if isinstance(env.action_space, gym.spaces.Discrete):
                    a_shape = (env.action_space.n,)
                else:
                    a_shape = env.action_space.shape
                s = env.reset()
                if 3 == len(s_shape):
                    s = img_preprocess(s)

                agent.set_up_task(test_task_id, s_shape, a_shape)

                temp_ep = 0
                check_ep = 0
                run_lim = test_run_lim_dict[test_task]
                run = 0
                total_runs = 0
                dt_start = datetime.now()
                ep_r = 0

                while total_runs < run_lim:
                    if render:
                        if "gridworld" == test_task:
                            env.render()
                            img = render_vals(env, agent, img, fig, ax, axbkgd)
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
                        if "CartPole-v1" == test_task:
                            ep_diff = temp_ep - check_ep
                            log_data = (total_runs, float(ep_r)/float(ep_diff+1), walltime.total_seconds())
                            check_ep = temp_ep
                        else:
                            log_data = (total_runs, float(ep_r)/float(run), walltime.total_seconds())
                        data_logger.store_test_data(log_data, test_task_id)
                        data_logger.save_test_data(test_task, test_task_id, idx)

                        print("TEST period r = {}".format(ep_r))

                        run = 0
                        ep_r = 0

                    if done or (total_runs == run_lim):
                        temp_ep += 1

                        print("TEST (ep {})".format(temp_ep))

                        s = env.reset()
                        if 3 == len(s_shape):
                            s = img_preprocess(s)

                data_logger.reset_test_data()

                env.close()

                if render:
                    plt.close('all')
    else:
        # TODO check

        agent.load()
        load_task_info()

        for idx, task in enumerate(task_arr):
            task_id = seen_task_dict[task]

            if "gridworld" == task:
                env = gridenv.GridEnvSim(10, 10, False, 1, False, True, 1)
                # TODO for final paper replace this w/ the harder (but more accepted) gym-minigrid
            else:
                env = gym.make(task)
                env.seed(seed_val+idx)

            s_shape = env.observation_space.shape
            a_shape = env.action_space.shape
            s = env.reset()
            if 3 == len(s_shape):
                s = img_preprocess(s)

            agent.set_up_task(task_id, s_shape, a_shape)

            done = False
            run = 0

            while not done and run < demo_runs_dict[task]:
                if render:
                    env.render()

                # get action
                a = agent.act(s, True)
                s_prime, r, done, _ = env.step(a)
                if 3 == len(s_shape):
                    s_prime = img_preprocess(s_prime)
                agent.store_experience(s, a, r, s_prime, done, -1, run, -1)
                s = np.copy(s_prime)
                run += 1

                # do data logging
                # TODO
                #log_data = ()
                #data_logger.store_demo_data(log_data)
                #data_logger.save_demo_data()

if __name__ == "__main__":
    main()
