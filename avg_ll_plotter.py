import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 14})

#data_dir = "./data/basic_seq_lrl_test"
data_dir = "./data/more-nodes_seq_lrl_test"
file_idxs = [1, 2, 3, 4, 5]
#exper_order = ["LunarLander-v2", "gridworld", "LunarLander-v2", "CartPole-v1", "gridworld", "CartPole-v1"] # TRAIN
exper_order = ["LunarLander-v2", "gridworld", "CartPole-v1"] # TEST
env_r_bounds = {"LunarLander-v2": [-10, 2], "gridworld": [-0.167, 0.0625], "CartPole-v1": [0, 500]}
colors = {"LunarLander-v2": "red", "gridworld": "blue", "CartPole-v1": "green"}
run_lim_dict = {"LunarLander-v2": 100000, "gridworld": 10000, "CartPole-v1": 10000}

#num_repeats = 2 # TRAIN
num_repeats = 1 # TEST

# TRAIN
#overall_episodes = {"LunarLander-v2_1": [()]*len(file_idxs), "gridworld_1": [()]*len(file_idxs), "LunarLander-v2_2": [()]*len(file_idxs), "CartPole-v1_1": [()]*len(file_idxs), "gridworld_2": [()]*len(file_idxs), "CartPole-v1_2": [()]*len(file_idxs)}
#overall_rs = {"LunarLander-v2_1": [()]*len(file_idxs), "gridworld_1": [()]*len(file_idxs), "LunarLander-v2_2": [()]*len(file_idxs), "CartPole-v1_1": [()]*len(file_idxs), "gridworld_2": [()]*len(file_idxs), "CartPole-v1_2": [()]*len(file_idxs)}
# TEST
overall_episodes = {"LunarLander-v2_1": [()]*len(file_idxs), "gridworld_1": [()]*len(file_idxs), "CartPole-v1_1": [()]*len(file_idxs)}
overall_rs = {"LunarLander-v2_1": [()]*len(file_idxs), "gridworld_1": [()]*len(file_idxs), "CartPole-v1_1": [()]*len(file_idxs)}

def rescale(val, in_min, in_max, out_min, out_max):
    return ((val - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min

for i in file_idxs:
    overall_ep_pickup = 0
    old_pickup_pts = {}
    pickup_pts = {}
    repeat_pts = {}
    env_switch_pts = []

    for idx, j in enumerate(exper_order):
        if j not in pickup_pts or j not in repeat_pts:
            old_pickup_pts[j] = 0
            pickup_pts[j] = 0
            repeat_pts[j] = 1
        old_pickup_pts[j] += pickup_pts[j]
        env_switch_pts.append(overall_ep_pickup)

        data_file = "TEST_" + str(i) + "_" + j + ".csv" # TRAIN
        #data_file = "TEST_TEST_" + str(i) + "_" + j + ".csv" # TEST
        df_data = pd.read_csv(os.path.join(data_dir, data_file), header=None, index_col=False)
    
        df_data = np.asarray(df_data.values.tolist())
        eps = df_data[:,0]
        if num_repeats == repeat_pts[j]:
            pickup_pts[j] = len(eps)+1
        else:
            pickup_pts[j] = np.where(eps==0)[0][repeat_pts[j]]
        rs = df_data[:,1]

        temp_eps = eps[old_pickup_pts[j]:pickup_pts[j]] + overall_ep_pickup
        temp_eps = temp_eps[:-2] # don't keep last episode
        temp_rs = rs[old_pickup_pts[j]:pickup_pts[j]]

        # NOTE activate for avg rs
        avg_rs = np.zeros_like(temp_rs)
        for k, x in enumerate(temp_rs):
            rs_subarr = temp_rs[max(0,k-100):k+1]
            mean_val = np.mean(rs_subarr)
            avg_rs[k] = mean_val
        temp_rs = avg_rs

        temp_rs = temp_rs[:-2]
        temp_rs = rescale(temp_rs, env_r_bounds[j][0], env_r_bounds[j][1], 0, 1)

        overall_episodes[j + "_" + str(repeat_pts[j])][i-1] = overall_episodes[j + "_" + str(repeat_pts[j])][i-1] + tuple(temp_eps)
        overall_rs[j + "_" + str(repeat_pts[j])][i-1] = overall_rs[j + "_" + str(repeat_pts[j])][i-1] + tuple(temp_rs)

        overall_ep_pickup += len(temp_eps)
        repeat_pts[j] += 1

episodes = []
r_data = []
r_data_stdev = []
episode_pickup = 0
overall_env_switch_pts = []
for k in overall_episodes:
    temp_ep_data = overall_episodes[k]
    temp_r_data = overall_rs[k]
    ep_cutoff = np.inf
    for t_ep in temp_ep_data:
        ep_cutoff = min(ep_cutoff, len(t_ep))
    episodes = np.append(episodes, np.arange(ep_cutoff)+episode_pickup)
    carryover_r_data = []
    for t_r in temp_r_data:
        carryover_r_data.append(t_r[:ep_cutoff])
    r_data = np.append(r_data, np.mean(carryover_r_data, 0))
    r_data_stdev = np.append(r_data_stdev, np.std(carryover_r_data, 0))
    overall_env_switch_pts.append(episode_pickup)
    episode_pickup = episode_pickup + ep_cutoff

fig, axs = plt.subplots(1, 1, figsize=(8,8))
axs.plot(episodes, r_data, linestyle='solid', color='black')
axs.fill_between(episodes, r_data-r_data_stdev/2., r_data+r_data_stdev/2., alpha=0.3, edgecolor='black', facecolor='black')
#axs.set_title("Performance")

#y_min, y_max = axs.get_ylim()
for idx, x_loc in enumerate(overall_env_switch_pts):
    col = colors[exper_order[idx]]
    #axs.axvline(x=x_loc, color=col)
    if idx < (len(overall_env_switch_pts)-1):
        axs.fill_between(episodes, 0, 1, where=(x_loc <= episodes) & (episodes < overall_env_switch_pts[idx+1]), color=col, alpha=0.2, transform=axs.get_xaxis_transform())
    else:
        axs.fill_between(episodes, 0, 1, where=(x_loc <= episodes) & (episodes < len(episodes)), color=col, alpha=0.2, transform=axs.get_xaxis_transform())

axs.set(xlabel='Episodes', ylabel='Averaged Reward')
#axs.xaxis.labelpad = -1
#axs.yaxis.labelpad = -1
#axs.text(0.01, -0.3, "(a)", transform=ax.transAxes)

exper_names = ["CartPole-v1", "gridworld", "LunarLander-v2"]
custom_lines = [Line2D([0], [0], color=colors[exper_names[0]]), Line2D([0], [0], color=colors[exper_names[1]]), Line2D([0], [0], color=colors[exper_names[2]])]
plt.legend(custom_lines, exper_names, loc="lower left", bbox_to_anchor=(0., 1.05, 1., 0.1), ncol=3, mode="expand", borderaxespad=0.)
#plt.subplots_adjust(wspace=0.225)
plt.savefig("TRAIN_avg.png", dpi=600, bbox_inches='tight')
#plt.savefig("TEST_avg.png", dpi=600, bbox_inches='tight')
#plt.show()
