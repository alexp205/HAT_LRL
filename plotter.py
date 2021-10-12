import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 14})

data_dir = "./data"
file_idxs = [1, 2, 3, 4, 5]
exper_names = ["CartPole-v1", "gridworld", "LunarLander-v2"] # must be unique

episodes = {}
runs = {}
r_data = [{}, {}] # [means, stdevs]
avg_r_data = [{}, {}] # [means, stdevs]
run_data = [{}, {}] # [means, stdevs]
time_data = [{}, {}] # [means, stdevs]

for idx, j in enumerate(exper_names):
    episodes[j] = []
    r_data[0][j] = []
    r_data[1][j] = []
    avg_r_data[0][j] = []
    avg_r_data[1][j] = []
    run_data[0][j] = []
    run_data[1][j] = []
    time_data[0][j] = []
    time_data[1][j] = []

    temp_episodes = []
    temp_r_data = []
    temp_avg_r_data = []
    temp_run_data = []
    temp_time_data = []

    min_episodes = np.inf

    for i in file_idxs:
        data_file = "lunar-first_" + str(i) + "_" + j + ".csv"
        df_data = pd.read_csv(os.path.join(data_dir, data_file), header=None, index_col=False)[:-1]

        df_data = np.asarray(df_data.values.tolist())
        #eps = df_data[:,0]
        min_episodes = min(len(df_data[:,0]), min_episodes)
        rs = df_data[:,1]
        avg_rs = np.zeros_like(rs)
        for i, x in enumerate(rs):
            rs_subarr = rs[max(0,i-100):i+1]
            mean_val = np.mean(rs_subarr)
            avg_rs[i] = mean_val
        runs = df_data[:,2]
        times = df_data[:,3]
        
        temp_r_data.append(rs)
        temp_avg_r_data.append(avg_rs)
        temp_run_data.append(runs)
        temp_time_data.append(times)

    temp_episodes = np.arange(min_episodes)
    for jdx, t_v in enumerate(temp_r_data):
        temp_r_data[jdx] = t_v[:min_episodes]
    for jdx, t_v in enumerate(temp_avg_r_data):
        temp_avg_r_data[jdx] = t_v[:min_episodes]
    for jdx, t_v in enumerate(temp_run_data):
        temp_run_data[jdx] = t_v[:min_episodes]
    for jdx, t_v in enumerate(temp_time_data):
        temp_time_data[jdx] = t_v[:min_episodes]

    episodes[j] = temp_episodes
    r_data[0][j] = np.mean(temp_r_data, 0)
    r_data[1][j] = np.std(temp_r_data, 0)
    avg_r_data[0][j] = np.mean(temp_avg_r_data, 0)
    avg_r_data[1][j] = np.std(temp_avg_r_data, 0)
    #cutoff_idx = np.argmax(avg_r_data[0][j])
    #avg_r_data[0][j] = avg_r_data[0][j][:cutoff_idx]
    #avg_r_data[1][j] = avg_r_data[1][j][:cutoff_idx]
    run_data[0][j] = np.mean(temp_run_data, 0)
    run_data[1][j] = np.std(temp_run_data, 0)
    time_data[0][j] = np.mean(temp_time_data, 0)
    time_data[1][j] = np.std(temp_time_data, 0)

fig, axs = plt.subplots(1, 3, figsize=(14,6), gridspec_kw={"width_ratios": [1,1,1]})
colors = ["red", "blue", "green", "purple"]
for idx, e_n in enumerate(exper_names):
    avg_r_means = avg_r_data[0][e_n]
    avg_r_stdevs = avg_r_data[1][e_n]
    axs[idx].plot(episodes[e_n], avg_r_means, color=colors[idx], linestyle='solid', label=e_n)
    axs[idx].fill_between(episodes[e_n], avg_r_means-avg_r_stdevs/2., avg_r_means+avg_r_stdevs/2., alpha=0.5, edgecolor=colors[idx], facecolor=colors[idx])
#axs[0].set_title("Performance")

custom_lines = [Line2D([0], [0], color=colors[0]), Line2D([0], [0], color=colors[1]), Line2D([0], [0], color=colors[2]), Line2D([0], [0], color=colors[3])]

axs[0].set(xlabel='Episodes', ylabel='Averaged Reward')
#axs[0].xaxis.labelpad = -1
#axs[0].yaxis.labelpad = -1
#axs[0].text(0.01, -0.3, "(a)", transform=ax.transAxes)

plt.legend(custom_lines, exper_names, loc="lower left", bbox_to_anchor=(-2.5, 1.05, 3., 0.1), ncol=len(exper_names), mode="expand", borderaxespad=0.)
#plt.subplots_adjust(wspace=0.225)
#plt.savefig('CHECK_ME.png', dpi=600, bbox_inches='tight')
plt.show()
