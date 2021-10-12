import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 14})

data_dir = "./data/init_test"
file_idxs = [1, 2, 3, 4, 5]
exper_order = ["LunarLander-v2", "gridworld", "LunarLander-v2", "CartPole-v1", "gridworld", "CartPole-v1"]
env_r_bounds = {"LunarLander-v2": [-10, 2], "gridworld": [-0.167, 0.0625], "CartPole-v1": [0, 500]}
colors = {"LunarLander-v2": "red", "gridworld": "blue", "CartPole-v1": "green"}

num_repeats = 2

def rescale(val, in_min, in_max, out_min, out_max):
    return ((val - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min

for i in file_idxs:
    episodes = []
    r_data = []

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

        data_file = "lunar-first_" + str(i) + "_" + j + ".csv"
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

        episodes = np.append(episodes, temp_eps)
        r_data = np.append(r_data, temp_rs)

        overall_ep_pickup += len(temp_eps)
        repeat_pts[j] += 1

    fig, axs = plt.subplots(1, 1, figsize=(8,8))
    axs.plot(episodes, r_data, linestyle='solid', color='black')
    #axs.set_title("Performance")

    #y_min, y_max = axs.get_ylim()
    for idx, x_loc in enumerate(env_switch_pts):
        col = colors[exper_order[idx]]
        #axs.axvline(x=x_loc, color=col)
        if idx < (len(env_switch_pts)-1):
            axs.fill_between(episodes, 0, 1, where=(x_loc <= episodes) & (episodes < env_switch_pts[idx+1]), color=col, alpha=0.2, transform=axs.get_xaxis_transform())
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
    plt.savefig("CHECK_ME" + str(i) + ".png", dpi=600, bbox_inches='tight')
    #plt.show()
