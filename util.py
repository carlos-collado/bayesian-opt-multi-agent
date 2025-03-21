# GENERAL PACKAGE IMPORT
# ----------------------

# PACKAGES
import numpy as np
import pandas as pd
import os
import random
import shutil
import math
from pyDOE2 import lhs
from mpi4py import MPI


# FUNCTIONS AND CLASSES OTHER FILES
from functions_to_sample import *

# MPI
comm = comm
rank = rank
size = size


# UTIL FUNCTIONS
# ----------------------
def transform_to_2d(an_array):
    if type(an_array) == list:
        an_array = np.array(an_array)
    if an_array.ndim == 1:
        an_array = np.expand_dims(an_array, axis=0)
    elif an_array.ndim == 0:
        an_array = np.atleast_2d(an_array)
    return an_array



def create_gifs(num_agents, num_funcs_per_agent, num_iterations):
    clear_directory('fig/GP_updates/gif')
    for i in range(num_agents):
        for j in range(num_funcs_per_agent):
            images = []
            for it in range(num_iterations):
                image_path = f'fig/GP_updates/png/gp_{it}.png'
                if os.path.exists(image_path):
                    images.append(imageio.imread(image_path))
            imageio.mimsave(f'fig/GP_updates/gif/gp_behavior.gif', images, fps=2)



def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')



def find_convergence_values(data):
    last_20_percent_index = int(0.8 * len(data))
    convergence_data = data[last_20_percent_index:]
    # Exclude any extreme outliers assuming they are transients and not affect the convergence value
    low_cutoff = np.percentile(convergence_data, 30)
    high_cutoff = np.percentile(convergence_data, 70)
    filtered_data = convergence_data[(convergence_data >= low_cutoff) & (convergence_data <= high_cutoff)]
    return np.mean(filtered_data), np.std(convergence_data)



def objective(fcn, noise = False, agent_id = 0, normalize = True):
    # random.seed(agent_id)
    noise_multiplier = 1 + noise * random.uniform(-0.2, 0.2)

    if fcn == 'boptest':
        return boptest_objective(noise_multiplier, normalize, agent_id)
    
    if fcn == 'fake_boptest':
        return f_objective(noise_multiplier)
    
    # if fcn == 'fake_boptest':
    #     return fake_boptest_objective(noise_multiplier)
    
    else:
        raise ValueError('Function not recognized')


def black_box_1(fcn, noise = False, agent_id = 0, num_bb = None, normalize = True):
    # random.seed(agent_id)
    noise_multiplier = 1 + noise * random.uniform(-0.1, 0.1)

    if fcn == 'boptest':
        return boptest_constraint(noise_multiplier, num_bb, normalize)
    
    if fcn == 'fake_boptest':
        return f_constraint(noise_multiplier, num_bb)

    # if fcn == 'fake_boptest':
    #     return fake_boptest_constraint(noise_multiplier, num_bb)
    
    else:
        raise ValueError('Function not recognized')


# def stratified_uniform_sample(x_range, x_dim=1, num_samples=100, discrete_num=10):
#     # Determine the discrete values within the range
#     # discrete_values = np.linspace(x_range[0], x_range[1], discrete_num, dtype=int)
#     discrete_values = np.linspace(x_range[0], x_range[1], discrete_num)
    
#     # Create all combinations of these discrete values for each dimension
#     mesh = np.meshgrid(*[discrete_values for _ in range(x_dim)])
#     all_combinations = np.vstack([m.ravel() for m in mesh]).T
    
#     # Shuffle to randomize order of combinations
#     np.random.shuffle(all_combinations)
    
#     # Ensure the number of samples doesn't exceed available unique combinations
#     if len(all_combinations) < num_samples:
#         raise ValueError("The number of unique samples requested exceeds the number of available unique combinations")
    
#     # Sample without replacement to ensure each combination is unique
#     sampled_indices = np.random.choice(len(all_combinations), size=num_samples, replace=False)
#     sampled_points = all_combinations[sampled_indices]

#     return sampled_points


def stratified_uniform_sample_lhs(x_range, x_dim=1, num_samples=100):
    # Generate the Latin Hypercube Samples
    samples = lhs(x_dim, samples=num_samples)
    
    # Scale samples to the specified range
    min_vals = np.full(x_dim, x_range[0])
    max_vals = np.full(x_dim, x_range[1])
    scaled_samples = min_vals + samples * (max_vals - min_vals)

    return scaled_samples



def cosine_annealing_lr(init_lr, total_iters):
    return lambda iteration: init_lr * (1 + math.cos(math.pi * iteration / total_iters)) / 2



def boptest_objective(noise_multiplier = 1, normalize = True, agent_id = 0):

    # comfort_df = base_temp_proflies_agent(agent_id)
    comfort_dfs_names = [f for f in os.listdir('comfort_80_files') if f.startswith('comf')]
    comfort_dfs_names = sorted(comfort_dfs_names, key=lambda x: int(x.split('_')[2]))

    if agent_id <= len(comfort_dfs_names):
        comfort_df = pd.read_csv(f'comfort_80_files/{comfort_dfs_names[agent_id]}')
        comfort_df['time'] = pd.DatetimeIndex(pd.to_datetime(comfort_df['time']))
        comfort_df = comfort_df.set_index('time')
        print(f'agent {agent_id}: {comfort_dfs_names[agent_id]}')

    else:
        comfort_df = base_temp_proflies_agent(agent_id)

    if normalize:
        T_dis_mean = 0
        T_dis_std = np.sqrt(boptest_variance(0, normalize=False))
    else:
        T_dis_mean = 0
        T_dis_std = 1

    def simulation(x):
        temp, _ = simulation_output(x)

        T_dis = calculate_discomfort(comfort_df, temp)

        T_dis = (T_dis - T_dis_mean) / T_dis_std

        return T_dis * noise_multiplier 

    return simulation



def boptest_constraint(noise_multiplier = 1, num_bb = None, normalize = True):

    if normalize:
        w_elec_mean = 0
        w_elec_std = np.sqrt(boptest_variance(4, normalize=False))
    else:
        w_elec_mean = 0
        w_elec_std = 1

    def simulation(x):
        _, welec_hour = simulation_output(x)
        w_elec = calculate_welec(welec_hour)

        w_elec = (w_elec - w_elec_mean) / w_elec_std

        return w_elec[num_bb-1] #* noise_multiplier   # as obj is k=0, constrs start from k=1

    return simulation



def f_objective(noise_multiplier = 1):
    def simulation(x):
        distances = [((np.array(x) - 19) ** 2).sum(), ((np.array(x) - 19) ** 2).sum()]
        min_distance = min(distances) * 0.04
        return min_distance * noise_multiplier
    return simulation

def f_constraint(noise_multiplier = 1, num_bb = None):
    mult_bb = [0.45, 0.5, 0.55]
    mult_bb[num_bb-1] = 1
    def simulation(x):
        return np.sum((np.array(x)-16) * mult_bb) * noise_multiplier * 0.25
    return simulation




# FUNCTION PARAMETERS
# ----------------------

def boptest_norm(k, fcn='boptest', normalize = False):

    noise_f_sample = False

    if fcn != 'boptest' and fcn != 'fake_boptest':
        raise ValueError('Function not implemented')
    
    if k == 0:
        '''
        Ensuring the person has the worst possible sensitivity (high ideal, small gap of acceptable values)
        a = []
        for i in range(1000):
            fo = objective(fcn = fcn, noise = False, agent_id = 0, normalize = True)
            a.append(fo(np.array([15, 15, 15, 15])))
        np.max(a)        
        '''
        if normalize:
            return 4100 / np.sqrt(boptest_variance(0, fcn, normalize = False))
        else:
            return 4100 # 4041.3117236945304
    
    elif k == 1:
        fc = black_box_1(fcn = fcn, noise = noise_f_sample, agent_id = 0, num_bb = 1, normalize = normalize)
        return fc(np.array([25, 15, 15]))

    elif k == 2:
        fc = black_box_1(fcn = fcn, noise = noise_f_sample, agent_id = 0, num_bb = 2, normalize = normalize)
        return fc(np.array([15, 25, 15]))

    elif k == 3:
        fc = black_box_1(fcn = fcn, noise = noise_f_sample, agent_id = 0, num_bb = 3, normalize = normalize)
        return fc(np.array([15, 15, 25]))
    
    elif k == 4:
        a = []
        for i in range(1, 4):
            a.append(boptest_norm(i, fcn=fcn, normalize = normalize))
        return np.max(a)

    else:
        raise ValueError('Function not implemented')





def boptest_mean(k, fcn='boptest', normalize = False, results = None):

    noise_f_sample = False

    if fcn != 'boptest' and fcn != 'fake_boptest':
        raise ValueError('Function not implemented')
    
    if results is None:
        with open(simulation_file, 'rb') as file:
            results = pickle.load(file)

    if k == 0:
        '''
        sample variance: as different discomfort for the same inputs (different people), need to sample multiple times:
        # Ensuring the person has the worst possible sensitivity (high ideal, gap of acceptable values)
        objs = []
        for i in tqdm(range(20)):
            fo = objective(fcn = fcn, noise = False, agent_id = 0)
            for x in tqdm(filtered_results.keys()):
                objs.append(fo(np.array(x)))
        np.mean(a)
        '''
        if normalize:
            return 250 / np.sqrt(boptest_variance(0, fcn, normalize = False))
        else:
            return 250 # 257.79648668639686

    elif k == 1:
        ''' 
        For the constraints, only need to sample once each input
        '''
        a = []
        fc = black_box_1(fcn = fcn, noise = noise_f_sample, agent_id = 0, num_bb = k, normalize = normalize)
        for x in results.keys():    
            a.append(fc(x))

        sample_mean = np.mean(a)
        return sample_mean

    elif k == 2:
        a = []
        fc = black_box_1(fcn = fcn, noise = noise_f_sample, agent_id = 0, num_bb = k, normalize = normalize)
        for x in results.keys():    
            a.append(fc(x))

        sample_mean = np.mean(a)
        return sample_mean
    
    elif k == 3:
        a = []
        fc = black_box_1(fcn = fcn, noise = noise_f_sample, agent_id = 0, num_bb = k, normalize = normalize)
        for x in results.keys():    
            a.append(fc(x))

        sample_mean = np.mean(a)
        return sample_mean
        
    elif k == 4:
        ''' 
        For the constraints, only need to sample once each input
        '''
        a = []
        for i in range(1, 4):
            fc = black_box_1(fcn = fcn, noise = noise_f_sample, agent_id = 0, num_bb = i, normalize = normalize)
            for x in results.keys():    
                a.append(fc(x))

        sample_mean = np.mean(a)
        return sample_mean
    
    else:
        raise ValueError('Function not implemented')





def boptest_variance(k, fcn='boptest', normalize = True, results = None, bounds = False):

    noise_f_sample = False

    if fcn != 'boptest' and fcn != 'fake_boptest':
        raise ValueError('Function not implemented')

    if results is None:
        with open(simulation_file, 'rb') as file:
            results = pickle.load(file)

    if k == 0:
        '''
        sample variance: as different discomfort for the same inputs (different people), need to sample multiple times:
        with open('filtered_results.pickle', 'rb') as file:
            results = pickle.load(file)

        sampled_keys = random.sample(list(results.keys()), 100)
        sampled_dict = {key: results[key] for key in sampled_keys}

        objs = []
        for i in tqdm(range(20)):
            fo = objective(fcn = fcn, noise = False, agent_id = 0, normalize=True)
            for x in sampled_dict.keys():
                objs.append(fo(np.array(x)))
        np.var(objs)
        '''
        if normalize:
            sample_variance = 35 # 36.669684903730094
        else:
            sample_variance = 125000 # 127464.09858927755

        if bounds:
            lower_bound = sample_variance * 1e-3
            upper_bound = (boptest_norm(k, fcn, normalize = normalize))**2
            return sample_variance, lower_bound, upper_bound
        else:
            return sample_variance

    elif k == 1:
        ''' 
        For the constraints, only need to sample once each input
        '''
        a = []
        fc = black_box_1(fcn = fcn, noise = noise_f_sample, agent_id = 0, num_bb = k, normalize=normalize)
        for x in results.keys():    
            a.append(fc(x))

        sample_variance = np.var(a)

    elif k == 2:
        a = []
        fc = black_box_1(fcn = fcn, noise = noise_f_sample, agent_id = 0, num_bb = k, normalize=normalize)
        for x in results.keys():    
            a.append(fc(x))

        sample_variance = np.var(a)
    
    elif k == 3:
        a = []
        fc = black_box_1(fcn = fcn, noise = noise_f_sample, agent_id = 0, num_bb = k, normalize=normalize)
        for x in results.keys():    
            a.append(fc(x))

        sample_variance = np.var(a)

    elif k == 4:

        if False:
            a = []
            for i in range(1, 4):
                fc = black_box_1(fcn = fcn, noise = noise_f_sample, agent_id = 0, num_bb = i, normalize = normalize)
                for x in results.keys():
                    a.append(fc(x))

            sample_variance = np.var(a)

        # ACCELERATE RUNNING
        # remove code above,
        # print(sample_variance)
        # and use this value
        sample_variance = 16385.873542726797

    else:
        raise ValueError('Function not implemented')
    
    if bounds:
        lower_bound = sample_variance * 1e-1
        upper_bound = (boptest_norm(k, fcn, normalize = normalize))**2
        return sample_variance, lower_bound, upper_bound

    else:
        return sample_variance






def boptest_noise_variance(k, fcn='boptest', normalize = True):

    _, lower_bound, _ = boptest_variance(k, fcn, normalize = normalize, bounds=True)
    norm = boptest_norm(k, fcn, normalize = normalize)

    noise_variance = 0.02 * norm
    lower_bound = 0.00001 * lower_bound
    upper_bound = 0.05 * norm
    return noise_variance, lower_bound, upper_bound




def boptest_legthscale(x_range, fcn='boptest', normalize = None):

    if fcn != 'boptest' and fcn != 'fake_boptest':
        raise ValueError('Function not implemented')

    range = x_range[1] - x_range[0]
    lengthscale = range
    lower_bound = 1e-5 * range
    upper_bound = 1e5 * range

    return lengthscale, lower_bound, upper_bound

