# ------------------------
# IMPORTS
# ------------------------
import os
import sys

# DEBUGGING
project_dir = '/home/ccollado/v15 DMABO paper local constraints v1'
os.chdir(project_dir)

# Add the directory to sys.path
sys.path.append(project_dir)

'''
# Print current directory and sys.path to debug
print("Current Directory:", os.getcwd())
print("Python Path:", sys.path)
print('-----------------------------')


# list files in directory
print('Files in directory:')
print(os.listdir())
print('-----------------------------')

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print('-----------------------------')
'''


from util import *
from functions_to_sample import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from tqdm import tqdm
from tabulate import tabulate
import scipy.ndimage

from IPython.display import display, clear_output
import time

import plotly.graph_objects as go
import plotly.express as px

import math

import GPy
import safeopt
import pickle

# Seeds
np.random.seed(42)
random.seed(42)


# ------------------------
# PARAMETERS
# ------------------------
max_blackbox = np.array([4895.10808108, 1985.46420108, 4561.61839914])


noise_f_sample = False
eta_value = 1e-1
run_horizon = 1000
num_instances = 1
plot_gif = False
num_samples = 850
beta = [1, 1] # try 1-3
num_agents = 10
fcn = 'boptest'
x_range = (16, 22)
x_dim = 3 #+ 1 # 4 low temp bound + 1 Kp
black_box_funcs_dim = 1 + 3 # objective + 3 time period constraints
discrete_num_per_dim = (x_range[1] - x_range[0]) * 5 + 1 # Integer values

suffix_fig = f'{run_horizon}it_{num_samples}sam_b{beta[0]}_alpha{eta_value}_maxsg12'
print(f"Suffix: {suffix_fig}")


# ------------------------
# RUN
# ------------------------

start = time.time()
config_instance = problemConfig(max_blackbox = max_blackbox, noise_f_sample = noise_f_sample, 
                                eta = eta_value, run_horizon = run_horizon, num_samples = num_samples, plot_gif= plot_gif,
                                beta = beta, num_agents = num_agents, fcn = fcn, x_range = x_range, x_dim = x_dim, discrete_num_per_dim = discrete_num_per_dim,
                                black_box_funcs_dim = black_box_funcs_dim
                                )

agent_lists, inputs, regrets, blacks, affines, lambdas, num_agents, num_black_box_constraints = run_instances(num_instances, config_instance)

end = time.time()

print(f"Total Time; {end - start} s")





# ------------------------
# CONVERGENCE PLOT
# ------------------------

max_power = config_instance.max_blackbox / np.sqrt(boptest_variance(4, normalize = False))
x_min = config_instance.x_range[0]
x_max = config_instance.x_range[1]


labels_legend = ['21:00-08:00', '16:00-21:00', '08:00-16:00']
colors = ['cornflowerblue', 'forestgreen', 'tomato']
max_colors = ['#1e3a5f', '#006400', '#8b2500']  # Dark versions of cornflower blue, light green, tomato


window_size = 10
transparency_real = 0.3
transparency_ewm = 1

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

fig, axs = plt.subplots(2, 2, figsize=(22, 14))  # Adjusted figsize to maintain proportion

# ------------------------------------------------
# Plot 1: Objective - Discomfort
# ------------------------------------------------
regrets_plot = regrets[num_samples:]

axs[0,0].plot(regrets_plot, color='crimson', alpha=transparency_real)
axs[0,0].plot(pd.Series(regrets_plot).ewm(span=window_size).mean(), color='crimson', label='Discomfort', alpha=transparency_ewm)
axs[0,0].set_xlim([0, len(regrets_plot)])
axs[0,0].set_title("Objective - Total Discomfort (°C h)")
axs[0,0].set_xlabel("Iteration")
axs[0,0].set_ylabel("Discomfort Level (°C h)")
axs[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=2)





# ------------------------------------------------
# Plot 2: Black-box Constraints
# ------------------------------------------------
blacks_plot = blacks[:, num_samples:]
total_bb_plot = blacks_plot.sum(axis=0)
peak_bb_plot = np.max(blacks_plot, axis=0)

for i, data in enumerate(blacks_plot):
    axs[0,1].plot(data, color=colors[i], alpha=transparency_real)  # Plot each data series with a label

for i, data in enumerate(blacks_plot):
    axs[0,1].plot(pd.Series(data).ewm(span=window_size).mean(), label=labels_legend[i], color=colors[i], alpha=transparency_ewm)

# Plot horizontal lines for max_power
for i, power in enumerate(max_power):
    axs[0,1].hlines(power, xmin=0, xmax=len(blacks_plot.T), colors='black', linestyles='dashed', alpha=0.7, label='Max Power ' + labels_legend[i], linewidth=3, color=max_colors[i])

axs[0,1].set_xlim([0, len(blacks_plot.T)])  # Set x-axis limits
axs[0,1].set_title("Black-box Constraints - Maximum Electricity Consumption (kWh)")
axs[0,1].set_xlabel("Iteration")
axs[0,1].set_ylabel("Electricity Consumption (kWh)")

# Collect handles and labels from the plot
handles, labels = axs[0,1].get_legend_handles_labels()

# Intercalate the handles and labels
intercalated_handles = []
intercalated_labels = []
num_entries = len(blacks_plot)
for i in range(num_entries):
    intercalated_handles.append(handles[i])  # Handle for blacks_plot
    intercalated_handles.append(handles[num_entries + i])  # Handle for corresponding max_power
    intercalated_labels.append(labels[i])  # Label for blacks_plot
    intercalated_labels.append(labels[num_entries + i])  # Label for corresponding max_power

# Plot the legend with intercalated labels
axs[0,1].legend(intercalated_handles, intercalated_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=3)





# ------------------------------------------------
# Plot 3: Lambdas
# ------------------------------------------------
lambdas_plot = np.array(lambdas[0]).T

for i, data in enumerate(lambdas_plot):
    axs[1,0].plot(data, color=colors[i], alpha=transparency_real)

for i, data in enumerate(lambdas_plot):
    axs[1,0].plot(pd.Series(data).ewm(span=window_size).mean(), label=labels_legend[i], color=colors[i], alpha=transparency_ewm)

axs[1,0].set_xlim([0, lambdas_plot.shape[1]])
axs[1,0].set_xlabel("Iteration")
axs[1,0].set_ylabel("Lambda, λ")
axs[1,0].set_title("Change of Dual Variables over Iterations")
axs[1,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=5)





# ------------------------------------------------
# Plot 4: Constraint Total
# ------------------------------------------------
gp_constr_plot = np.sum([np.array(agent.constrain_lcb, dtype=float) for agent in agent_lists[0]], axis=0).T
peak_gp_bb_plot = np.max(gp_constr_plot, axis=0)

for i, data in enumerate(gp_constr_plot):
    axs[1,1].plot(data, color=colors[i], alpha=transparency_real)  # Label for gp_constr_plot data

for i, data in enumerate(gp_constr_plot):
    axs[1,1].plot(pd.Series(data).ewm(span=window_size).mean(), label=labels_legend[i], color=colors[i], alpha=transparency_ewm)

# Setting x limits for the plot
axs[1,1].set_xlim([0, len(gp_constr_plot.T)])

# Plotting each max_power with corresponding label
for i, power in enumerate(max_power):
    axs[1,1].hlines(power, xmin=0, xmax=len(gp_constr_plot.T), colors='black', linestyles='dashed', alpha=1, label='Max Power ' + labels[i], linewidth=3, color=max_colors[i])

axs[1,1].set_title("Gaussian Process Black-box Constraints - Max Welec")
axs[1,1].set_xlabel("Iteration")
axs[1,1].set_ylabel("Welec Value")

handles, labels = axs[1,1].get_legend_handles_labels()
intercalated_handles = []
intercalated_labels = []
num_entries = len(max_power)  # Assuming equal length for gp_constr_plot and max_power arrays
for i in range(num_entries):
    intercalated_handles.append(handles[i])  # Handle for gp_constr_plot
    intercalated_handles.append(handles[num_entries + i])  # Handle for corresponding max_power
    intercalated_labels.append(labels[i])  # Label for gp_constr_plot
    intercalated_labels.append(labels[num_entries + i])  # Label for corresponding max_power
axs[1,1].legend(intercalated_handles, intercalated_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=3)


plt.tight_layout()
plt.savefig(f'/home/ccollado/v15 DMABO paper local constraints v1/run_script/summary_{suffix_fig}.png')
plt.close()




# ------------------------
# INPUTS PLOT
# ------------------------

inputs_plot = inputs.copy().iloc[num_samples:].reset_index(drop=True)

# Split the tuples into individual columns for each agent
for agent in inputs_plot.columns:
    for input_index in range(x_dim):
        inputs_plot[(agent, f'Input_{input_index+1}')] = inputs_plot[agent].apply(lambda x: x[input_index])

# # Drop the original columns with tuples
inputs_plot = inputs_plot.drop(columns=list(range(num_agents)))

# # Reorganize the columns to better format the DataFrame
inputs_plot.columns = pd.MultiIndex.from_tuples(inputs_plot.columns)
inputs_plot = inputs_plot.sort_index(axis=1)

window_size = 20
transparency_real = 0.3
transparency_ewm = 1

num_agents = len(inputs_plot.columns.levels[0])  # Number of unique agents
num_cols = 2
num_rows = (num_agents + 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()  # Flatten to simplify indexing

# Collect handles and labels for the legend
handles, labels = [], []

# Plotting
for i in range(num_agents):
    for j, input_key in enumerate(inputs_plot.columns.levels[1]):  # Iterate over each type of input
        line, = axes[i].plot(inputs_plot[(i, input_key)], label='Max Power ' + labels_legend[j], color=colors[j], alpha=transparency_real)

    for j, input_key in enumerate(inputs_plot.columns.levels[1]):  # Iterate over each type of input
        line, = axes[i].plot(inputs_plot[(i, input_key)].ewm(span=window_size).mean(), label='Max Power ' + labels_legend[j], color=colors[j], alpha=transparency_ewm)
        if i == 0:  # Only add one set of handles/labels to the legend
            handles.append(line)
            labels.append('Max Power ' + labels_legend[j])

    axes[i].set_title(f'Agent {i}')
    axes[i].set_xlabel('Iteration')
    axes[i].set_ylabel('Value (°C)')
    axes[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    axes[i].set_xlim([0, len(inputs_plot)])
    axes[i].set_ylim([x_range[0]-0.1, x_range[1]+0.1])

# Turn off unused subplots if any
for ax in axes[num_agents:]:
    ax.axis('off')

# Place a common legend at the top
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=len(labels))

plt.tight_layout()
plt.savefig(f'/home/ccollado/v15 DMABO paper local constraints v1/run_script/inputs_{suffix_fig}.png')
plt.close()

print('INPUTS')
print(inputs_plot.iloc[-1].values)

print('LAMBDAS')
print(lambdas[0][-3:])


print("Done!")