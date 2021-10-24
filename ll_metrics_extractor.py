import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import re

is_indiv = True

plt.rcParams.update({'font.size': 14})

#data_dir = "./data/more-layers_seq_lrl_test"
#data_dir = "./data/naive_baselines"
data_dir = "./data/indiv_baselines"
file_idxs = [1, 2, 3, 4, 5]
exper_order = ["LunarLander-v2", "gridworld", "CartPole-v1"] # TEST
env_r_bounds = {"LunarLander-v2": [-10, 2], "gridworld": [-0.167, 0.0625], "CartPole-v1": [0, 500]}

def rescale(val, in_min, in_max, out_min, out_max):
    return ((val - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min

if not is_indiv:
    task_results = {"LunarLander-v2": np.full((len(file_idxs), len(exper_order)), -1.), "gridworld": np.full((len(file_idxs), len(exper_order)), -1.), "CartPole-v1": np.full((len(file_idxs), len(exper_order)), -1.)}

    for i in file_idxs:
        for j in range(len(exper_order)):
            #query_str_seed = "TEST_layers_{}".format(i)
            query_str_seed = "TEST_naive_{}".format(i)
            query_str_iter = "iter_{}".format(j)
    
            test_files = [f for f in os.listdir(data_dir) if ((query_str_seed in f) and (query_str_iter in f))]
            for t_f in test_files:
                df_data = pd.read_csv(os.path.join(data_dir, t_f), header=None, index_col=False)
                exper_name = re.search('iter_{}_(.+?).csv'.format(j), t_f).group(1)
        
                df_data = np.asarray(df_data.values.tolist())
                rs = df_data[:,1]
    
                # NOTE activate for avg rs
                avg_rs = np.zeros_like(rs)
                for k, x in enumerate(rs):
                    rs_subarr = rs[max(0,k-100):k+1]
                    mean_val = np.mean(rs_subarr)
                    avg_rs[k] = mean_val
                rs = avg_rs
                rs = rs[:-2]
                rs = rescale(rs, env_r_bounds[exper_name][0], env_r_bounds[exper_name][1], 0, 1)
    
                avg_r = np.mean(rs)
    
                #print("t_f: {}".format(t_f))
                #print("avg perf: {}".format(avg_r))
    
                task_results[exper_name][i-1][j] = avg_r
    
    ll_test_matrix = np.zeros((3,3))
    ll_test_matrix_std = np.zeros((3,3))
    for idx, exper_name in enumerate(exper_order):
        exper_results_raw = task_results[exper_name]
        exper_results = np.mean(exper_results_raw, axis=0)
        exper_results_std = np.std(exper_results_raw, axis=0)
    
        for jdx, e_r in enumerate(exper_results):
            if e_r >= 0:
                ll_test_matrix[idx][jdx] = e_r
                ll_test_matrix_std[idx][jdx] = exper_results_std[jdx]
    
    print(ll_test_matrix)
    print(ll_test_matrix_std)
    
    ll_results = np.empty(ll_test_matrix.shape, dtype=object)
    for idx, a in enumerate(ll_test_matrix):
        for jdx, b in enumerate(a):
            ll_results[idx][jdx] = "{0} ({1})".format(round(b,3), round(ll_test_matrix_std[idx][jdx],3))
    
    print(ll_results)
    
    fig, ax = plt.subplots()
    ax.axis('off')
    
    col_labels = ["Task 1", "Task 2", "Task 3"]
    table = ax.table(cellText=ll_results, cellLoc='center', rowLabels=exper_order, colLabels=col_labels, loc='center')
    table.scale(1,3)
else:
    task_results = {"LunarLander-v2": np.full((len(file_idxs)), -1.), "gridworld": np.full((len(file_idxs)), -1.), "CartPole-v1": np.full((len(file_idxs)), -1.)}

    for i in file_idxs:
        for j in exper_order:
            query_str = "TEST_indiv_{0}_{1}".format(i,j)
    
            test_files = [f for f in os.listdir(data_dir) if query_str in f]
            for t_f in test_files:
                df_data = pd.read_csv(os.path.join(data_dir, t_f), header=None, index_col=False)
        
                df_data = np.asarray(df_data.values.tolist())
                rs = df_data[:,1]
    
                # NOTE activate for avg rs
                avg_rs = np.zeros_like(rs)
                for k, x in enumerate(rs):
                    rs_subarr = rs[max(0,k-100):k+1]
                    mean_val = np.mean(rs_subarr)
                    avg_rs[k] = mean_val
                rs = avg_rs
                rs = rs[:-2]
                rs = rescale(rs, env_r_bounds[j][0], env_r_bounds[j][1], 0, 1)
    
                avg_r = np.mean(rs)
    
                #print("t_f: {}".format(t_f))
                #print("avg perf: {}".format(avg_r))
    
                task_results[j][i-1] = avg_r
    
    ll_test_matrix = np.zeros((3,1))
    ll_test_matrix_std = np.zeros((3,1))
    for idx, exper_name in enumerate(exper_order):
        exper_results_raw = task_results[exper_name]
        exper_results = np.mean(exper_results_raw)
        exper_results_std = np.std(exper_results_raw)
    
        ll_test_matrix[idx] = exper_results
        ll_test_matrix_std[idx][0] = exper_results_std
    
    print(ll_test_matrix)
    print(ll_test_matrix_std)
    
    ll_results = np.empty(ll_test_matrix.shape, dtype=object)
    for idx, a in enumerate(ll_test_matrix):
        #ll_results[idx] = "{0} ({1})".format(round(a,3), round(ll_test_matrix_std[idx],3))
        ll_results[idx] = "{:.3f} ({:.3f})".format(a[0], ll_test_matrix_std[idx][0])
    
    print(ll_results)
    
    fig, ax = plt.subplots()
    ax.axis('off')
    
    table = ax.table(cellText=ll_results, cellLoc='center', rowLabels=exper_order, loc='center')
    table.scale(1,3)

#plt.show()
#plt.savefig("ll_results.png", dpi=600, bbox_inches='tight')
#plt.savefig("ll_naive_results.png", dpi=600, bbox_inches='tight')
plt.savefig("ll_indiv_results.png", dpi=600, bbox_inches='tight')
