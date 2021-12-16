import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 14})

data_dir = "./data/nodeslayers_seq_lrl_test"
naive_data_dir = "./data/naive_baselines"
indiv_data_dir = "./data/indiv_baselines"
file_idxs = [1, 2, 3, 4, 5]
#exper_order = ["LunarLander-v2", "gridworld", "LunarLander-v2", "CartPole-v1", "gridworld", "CartPole-v1"] # TRAIN
exper_order = ["LunarLander-v2", "gridworld", "CartPole-v1"] # TEST
env_r_bounds = {"LunarLander-v2": [-9, 2], "gridworld": [-0.169, 0.063], "CartPole-v1": [0, 500]}
colors = {"LunarLander-v2": "red", "gridworld": "blue", "CartPole-v1": "green"}

#num_repeats = 2 # TRAIN
num_repeats = 1 # TEST

def rescale(val, in_min, in_max, out_min, out_max):
    return ((val - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min

for i in file_idxs:
    episodes = []
    r_data = []
    naive_episodes = []
    naive_r_data = []
    indiv_episodes = []
    indiv_r_data = []

    overall_ep_pickup = 0
    old_pickup_pts = {}
    pickup_pts = {}
    repeat_pts = {}
    env_switch_pts = []
    naive_overall_ep_pickup = 0
    naive_old_pickup_pts = {}
    naive_pickup_pts = {}
    naive_repeat_pts = {}
    naive_env_switch_pts = []
    indiv_overall_ep_pickup = 0
    indiv_old_pickup_pts = {}
    indiv_pickup_pts = {}
    indiv_repeat_pts = {}
    indiv_env_switch_pts = []

    for idx, j in enumerate(exper_order):
        if j not in pickup_pts or j not in repeat_pts:
            old_pickup_pts[j] = 0
            pickup_pts[j] = 0
            repeat_pts[j] = 1
        old_pickup_pts[j] += pickup_pts[j]
        env_switch_pts.append(overall_ep_pickup)
        if j not in naive_pickup_pts or j not in naive_repeat_pts:
            naive_old_pickup_pts[j] = 0
            naive_pickup_pts[j] = 0
            naive_repeat_pts[j] = 1
        naive_old_pickup_pts[j] += naive_pickup_pts[j]
        naive_env_switch_pts.append(naive_overall_ep_pickup)
        if j not in indiv_pickup_pts or j not in indiv_repeat_pts:
            indiv_old_pickup_pts[j] = 0
            indiv_pickup_pts[j] = 0
            indiv_repeat_pts[j] = 1
        indiv_old_pickup_pts[j] += indiv_pickup_pts[j]
        indiv_env_switch_pts.append(indiv_overall_ep_pickup)

        data_file = "TEST_nodeslayers_" + str(i) + "_iter_2_" + j + ".csv" # TRAIN
        df_data = pd.read_csv(os.path.join(data_dir, data_file), header=None, index_col=False)
        naive_data_file = "TEST_naive_" + str(i) + "_iter_2_" + j + ".csv" # TRAIN
        naive_df_data = pd.read_csv(os.path.join(naive_data_dir, naive_data_file), header=None, index_col=False)
        indiv_data_file = "TEST_indiv_" + str(i) + "_iter_0_" + j + ".csv" # TRAIN
        indiv_df_data = pd.read_csv(os.path.join(indiv_data_dir, indiv_data_file), header=None, index_col=False)
    
        df_data = np.asarray(df_data.values.tolist())
        eps = df_data[:,0]
        if num_repeats == repeat_pts[j]:
            pickup_pts[j] = len(eps)+1
        else:
            pickup_pts[j] = np.where(eps==0)[0][repeat_pts[j]]
        rs = df_data[:,2]
        naive_df_data = np.asarray(naive_df_data.values.tolist())
        naive_eps = naive_df_data[:,0]
        if num_repeats == naive_repeat_pts[j]:
            naive_pickup_pts[j] = len(naive_eps)+1
        else:
            naive_pickup_pts[j] = np.where(naive_eps==0)[0][naive_repeat_pts[j]]
        naive_rs = naive_df_data[:,2]
        indiv_df_data = np.asarray(indiv_df_data.values.tolist())
        indiv_eps = indiv_df_data[:,0]
        if num_repeats == indiv_repeat_pts[j]:
            indiv_pickup_pts[j] = len(indiv_eps)+1
        else:
            indiv_pickup_pts[j] = np.where(indiv_eps==0)[0][indiv_repeat_pts[j]]
        indiv_rs = indiv_df_data[:,2]

        temp_eps = eps[old_pickup_pts[j]:pickup_pts[j]] + overall_ep_pickup
        temp_rs = rs[old_pickup_pts[j]:pickup_pts[j]]
        naive_temp_eps = naive_eps[naive_old_pickup_pts[j]:naive_pickup_pts[j]] + naive_overall_ep_pickup
        naive_temp_rs = naive_rs[naive_old_pickup_pts[j]:naive_pickup_pts[j]]
        indiv_temp_eps = indiv_eps[indiv_old_pickup_pts[j]:indiv_pickup_pts[j]] + indiv_overall_ep_pickup
        indiv_temp_rs = indiv_rs[indiv_old_pickup_pts[j]:indiv_pickup_pts[j]]

        # NOTE activate for avg rs
        avg_rs = np.zeros_like(temp_rs)
        for k, x in enumerate(temp_rs):
            rs_subarr = temp_rs[max(0,k-100):k+1]
            mean_val = np.mean(rs_subarr)
            avg_rs[k] = mean_val
        temp_rs = avg_rs
        naive_avg_rs = np.zeros_like(naive_temp_rs)
        for k, x in enumerate(naive_temp_rs):
            rs_subarr = naive_temp_rs[max(0,k-100):k+1]
            mean_val = np.mean(rs_subarr)
            naive_avg_rs[k] = mean_val
        naive_temp_rs = naive_avg_rs
        indiv_avg_rs = np.zeros_like(indiv_temp_rs)
        for k, x in enumerate(indiv_temp_rs):
            rs_subarr = indiv_temp_rs[max(0,k-100):k+1]
            mean_val = np.mean(rs_subarr)
            indiv_avg_rs[k] = mean_val
        indiv_temp_rs = indiv_avg_rs

        temp_rs = rescale(temp_rs, env_r_bounds[j][0], env_r_bounds[j][1], 0, 1)
        naive_temp_rs = rescale(naive_temp_rs, env_r_bounds[j][0], env_r_bounds[j][1], 0, 1)
        indiv_temp_rs = rescale(indiv_temp_rs, env_r_bounds[j][0], env_r_bounds[j][1], 0, 1)

        episodes = np.append(episodes, temp_eps)
        r_data = np.append(r_data, temp_rs)
        naive_episodes = np.append(naive_episodes, naive_temp_eps)
        naive_r_data = np.append(naive_r_data, naive_temp_rs)
        indiv_episodes = np.append(indiv_episodes, indiv_temp_eps)
        indiv_r_data = np.append(indiv_r_data, indiv_temp_rs)

        overall_ep_pickup += len(temp_eps)
        repeat_pts[j] += 1
        naive_overall_ep_pickup += len(naive_temp_eps)
        naive_repeat_pts[j] += 1
        indiv_overall_ep_pickup += len(indiv_temp_eps)
        indiv_repeat_pts[j] += 1

    fig, axs = plt.subplots(1, 1, figsize=(8,8))
    exper_line, = axs.plot(episodes, r_data, linestyle='solid', color='black', label="HAT")
    naive_line, = axs.plot(naive_episodes, naive_r_data, linestyle='dashed', color='black', label="Naive")
    indiv_line, = axs.plot(indiv_episodes, indiv_r_data, linestyle='dashdot', color='black', label="Base")
    #axs.set_title("Performance")

    #print(env_switch_pts)

    #y_min, y_max = axs.get_ylim()
    for idx, x_loc in enumerate(env_switch_pts):
        col = colors[exper_order[idx]]
        #axs.axvline(x=x_loc, color=col)
        if idx < (len(env_switch_pts)-1):
            axs.fill_between(episodes, 0, 1, where=(x_loc <= episodes) & (episodes < env_switch_pts[idx+1]), color=col, alpha=0.2, transform=axs.get_xaxis_transform())
        else:
            axs.fill_between(episodes, 0, 1, where=(x_loc <= episodes) & (episodes < len(episodes)), color=col, alpha=0.2, transform=axs.get_xaxis_transform())
            break
    
    axs.set(xlabel='Episodes', ylabel='Averaged Reward')
    #axs.xaxis.labelpad = -1
    #axs.yaxis.labelpad = -1
    #axs.text(0.01, -0.3, "(a)", transform=ax.transAxes)
    
    exper_names = ["LunarLander-v2", "gridworld", "CartPole-v1"]
    custom_lines = [Line2D([0], [0], color=colors[exper_names[0]]), Line2D([0], [0], color=colors[exper_names[1]]), Line2D([0], [0], color=colors[exper_names[2]])]
    task_color_legend = plt.legend(custom_lines, exper_names, loc="lower left", bbox_to_anchor=(0., 1.05, 1., 0.1), ncol=3, mode="expand", borderaxespad=0.)
    plt.gca().add_artist(task_color_legend)
    plt.legend(handles=[exper_line, naive_line, indiv_line], loc="lower right")
    #plt.subplots_adjust(wspace=0.225)
    #plt.savefig("TRAIN_indiv_" + str(i) + ".png", dpi=600, bbox_inches='tight')
    #plt.savefig("HAT_TEST_results_" + str(i) + ".png", dpi=600, bbox_inches='tight')
    plt.show()
