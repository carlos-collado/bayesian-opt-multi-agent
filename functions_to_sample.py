# GENERAL PACKAGE IMPORT
# ----------------------
# import requests
import sys
import numpy as np
from custom_kpi.custom_kpi_calculator import CustomKPI
from controllers.controller import Controller
import json
import collections
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
import random
import pickle
from matplotlib import pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from mpi4py import MPI

PLOT_PROFILE = False
INTERPOLATE_FUNC = True
N_PERIODS = 3

simulation_file = 'input_files/simulation_3_periods.pickle'


# ----------------------------------------------------------------------------------------
# MPI
# ----------------------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



# -*- coding: utf-8 -*-
"""
This module is an example python-based testing interface.  It uses the
``requests`` package to make REST API calls to the test case container,
which must already be running.  A controller is tested, which is
imported from a different module.

"""

# ----------------------------------------------------------------------------------------
# TESTCASE=bestest_hydronic_heat_pump docker-compose up
# docker-compose down
# ----------------------------------------------------------------------------------------

def categorize_time(hour):

    if N_PERIODS == 3:
        if 21 <= hour or hour < 8:
            return 0  # Off-peak hours from 9:00 PM to 8:00 AM
        elif 16 <= hour < 21:
            return 1  # Peak hours from 4:00 PM to 9:00 PM
        else:
            return 2  # Normal hours from 8:00 AM to 4:00 PM
        
    elif N_PERIODS == 4:
        return int(np.floor(hour / 6))  # 4 periods of 6 hours each
    
    elif N_PERIODS == 1:
        return 0



# CHECK RESPONSE
# ----------------------
def check_response(response):
    """Check status code from restful API and return result if no error

    Parameters
    ----------

    response: obj, response object

    Returns
    -------
    result : dict, result from call to restful API

    """
    if isinstance(response, requests.Response):
        status = response.status_code
    if status == 200:
        response = response.json()['payload']
        return response
    print("Unexpected error: {}".format(response.text))
    print("Exiting!")
    sys.exit()


# CONTROL TEST
# ----------------------
def control_test(control_module='', start_time=0, warmup_period=5*24*3600, length=24*3600, 
                       scenario=None, step=300, customized_kpi_config=None, use_forecast=False,
                       Kp=None, LowerSetp_list=None, warmup_days=None):

    # SETUP TEST
    # -------------------------------------------------------------------------
    url = 'http://127.0.0.1:5000'
    controller = Controller(control_module, use_forecast)

    # GET TEST INFORMATION
    # -------------------------------------------------------------------------
    inputs = check_response(requests.get('{0}/inputs'.format(url)))
    measurements = check_response(requests.get('{0}/measurements'.format(url)))

    # RUN TEST CASE
    # -------------------------------------------------------------------------
    # Initialize test with a specified start time and warmup period
    res = check_response(requests.put('{0}/initialize'.format(url), json={'start_time': start_time, 'warmup_period': warmup_period}))
    # Set final time and total time steps according to specified length (seconds)
    final_time = start_time + length
    total_time_steps = int(length / step)  # calculate number of timesteps
    # Set simulation time step
    res = check_response(requests.put('{0}/step'.format(url), json={'step': step}))
    # Initialize input to simulation from controller
    u = controller.initialize()
    # Simulation Loop
    for t in range(total_time_steps):
        # Advance simulation with control input value(s)
        y = check_response(requests.post('{0}/advance'.format(url), json=u))
        # If simulation is complete break simulation loop
        if not y:
            break
        # Compute control signal input to simulation for the next timestep
        hour_of_day = int(np.floor(y['time'] / 3600 % 24))  # Convert seconds to hour
        period_day = categorize_time(hour_of_day)

        # period_day = int(np.floor(y['time'] / 3600 % 24 / 6))
        LowerSetp = LowerSetp_list[period_day]
        u = controller.compute_control(y, Kp, LowerSetp)
    # VIEW RESULTS
    # -------------------------------------------------------------------------

    points = list(measurements.keys()) + list(inputs.keys())
    df_res = pd.DataFrame()
    res = check_response(requests.put('{0}/results'.format(url), json={'point_names': points, 'start_time': start_time, 'final_time': final_time}))
    df_res = pd.DataFrame.from_dict(res)
    df_res = df_res.set_index('time')

    df_res.drop(df_res.index[-1], inplace=True)
    df_res = df_res.copy().iloc[120*24*warmup_days:] # Remove first __ days (warmup)

    # Temp profile
    temp_profile = df_res.copy()
    temp_profile.index = pd.to_datetime(temp_profile.index, unit='s', origin='unix')
    temp_profile = pd.DataFrame(temp_profile.resample('1h').mean()['reaTZon_y'])

    day_number = temp_profile.index.dayofyear - temp_profile.index.dayofyear.min()
    hour_of_day = temp_profile.index.hour
    multi_index = pd.MultiIndex.from_arrays([day_number, hour_of_day], names=['day', 'hour'])
    temp_profile.index = multi_index
    temp_profile = temp_profile - 273.15
    temp_profile = temp_profile['reaTZon_y'].values

    # Welec
    w_elec = df_res['reaPHeaPum_y'].copy()
    w_elec.index = (np.floor((df_res.index) / 3600 % 24)).astype(int)
    welec_hour = w_elec.groupby(w_elec.index).sum()  / 1000 * 30 / 3600 # Convert from W to kWh
    welec_hour = welec_hour.values

    return temp_profile, welec_hour



# def simulation_output(inputs):
#     tuple_inputs = tuple(inputs)

#     with open('simulation_results_copy.pickle', 'rb') as file:
#         results = pickle.load(file)

#     if tuple_inputs not in results.keys():

#         print('Simulation not found: ', tuple_inputs)

#         temp_profile, welec_hour = control_test(control_module='controllers.pid', length=40*24*3600, step=3600, Kp=1, 
#                                         LowerSetp_list=tuple(inputs+273.15), warmup_period=0*24*3600, start_time=0*24*3600,
#                                     warmup_days = 10)

#         results[tuple_inputs] = {'welec': welec_hour, 'temp': temp_profile}

#         with open('simulation_results_copy.pickle', 'wb') as f:
#             pickle.dump(results, f)
#             print('Simulation saved: ', tuple_inputs)

#     else:

#         temp_profile = results[tuple_inputs]['temp']
#         welec_hour = results[tuple_inputs]['welec']

#     return temp_profile, welec_hour




def simulation_output(inputs):
    tuple_inputs = tuple(inputs)

    with open(simulation_file, 'rb') as file:
        simulation = pickle.load(file)

    if tuple_inputs not in simulation.keys():

        if INTERPOLATE_FUNC:
            # print('Interpolating Results: ', tuple_inputs)

            keys = np.array(list(simulation.keys()))
            welec_values = np.vstack([v['welec'] for v in simulation.values()])
            temp_values = np.vstack([v['temp'] for v in simulation.values()])

            interp_welec = LinearNDInterpolator(keys, welec_values)
            interp_temp = LinearNDInterpolator(keys, temp_values)

            # Interpolate values
            temp_profile = interp_temp(tuple_inputs)
            welec_hour = interp_welec(tuple_inputs)

        else:

            print('Simulation not found: ', tuple_inputs)

            temp_profile, welec_hour = control_test(control_module='controllers.pid', length=40*24*3600, step=3600, Kp=1, 
                                            LowerSetp_list=tuple(inputs+273.15), warmup_period=0*24*3600, start_time=0*24*3600,
                                        warmup_days = 10)

            simulation[tuple_inputs] = {'welec': welec_hour, 'temp': temp_profile}

            with open(simulation_file, 'wb') as f:
                pickle.dump(simulation, f)
                print('Simulation saved: ', tuple_inputs)

    else:
        # Directly retrieve values
        temp_profile = simulation[tuple_inputs]['temp']
        welec_hour = simulation[tuple_inputs]['welec']

    return temp_profile, welec_hour




# RUN DMABO
# -------------------------------------------------------------------------
class problemConfig():
    def __init__(self, problem_name = 'power_allocation_agent', x_dim = 1, 
                 x_range = (0, 5), black_box_funcs_dim = 2, discrete_num_per_dim = 51,
                 num_samples = 5, bb_noise_var = 0.02 ** 2, eta = 0.0005, beta = 1,
                 num_agents = 2, run_horizon = 100,
                 max_affine = 1e10, max_blackbox = 0.9, epsilon = 1e-3,
                 noise_f_sample = False, fcn = 'f', 
                 plot_gif = False):

        # Problem Dimensions and Ranges
        self.problem_name = problem_name
        self.x_dim = x_dim
        self.x_range = x_range
        self.black_box_funcs_dim = black_box_funcs_dim
        self.discrete_num_per_dim = discrete_num_per_dim

        # Sampling and Noise
        self.num_samples = num_samples
        self.bb_noise_var = bb_noise_var

        # Hyperparameters
        self.eta = eta
        self.beta = beta

        # Instance Parameters
        self.num_agents = num_agents
        self.run_horizon = run_horizon

        # Constraints
        self.max_affine = max_affine
        self.max_blackbox = max_blackbox
        self.epsilon = epsilon

        # Function to sample
        self.noise_f_sample = noise_f_sample
        self.fcn = fcn # 'f' or 'boptest'

        # Plotting
        self.plot_gif = plot_gif




def DMABO(config_instance):
    from dmabo_instance import PowerAllocationInstance

    pa_inst = PowerAllocationInstance(config_instance)
    random_coordinator, lambda_list, mu_list = pa_inst.run_one_instance()

    agent_list = random_coordinator.local_agents
    num_agent = config_instance.num_agents
    num_local_agents = len(agent_list)

    obj = sum(agent_list[k].black_box_func_gp_list[0].Y for k in range(num_local_agents))
    constr_affine = sum(agent_list[k].black_box_func_gp_list[0].X for k in range(num_local_agents)) - random_coordinator.max_affine
    obj_total = comm.allreduce(obj, op=MPI.SUM)
    constr_affine_total = comm.allreduce(constr_affine, op=MPI.SUM)
    opt_val = 0
    total_regret = obj_total - opt_val

    if(len(agent_list[0].black_box_func_gp_list)) > 1:
        constr_black = np.array([sum([agent_list[k].black_box_func_gp_list[m+1].Y for k in range(num_local_agents)]).reshape(-1) for m in range(len(agent_list[0].black_box_func_gp_list)-1)])
    else: 
        constr_black = np.array((np.nan))

    constr_black_total = comm.allreduce(constr_black, op=MPI.SUM)

    # Initialize the inputs array to hold objects (tuples).
    inputs = np.empty((len(agent_list[0].black_box_func_gp_list[0].X), num_agent), dtype=object)
    idx = 0
    for k in range(num_agent):
        if k >= random_coordinator.start_index and k < random_coordinator.end_index:
            # print('Rank: ', rank, 'Agent: ', k, 'Start Index: ', random_coordinator.start_index, 'End Index: ', random_coordinator.end_index, 'Local Agents: ', num_local_agents, 'Length in_array: ', len(agent_list[idx].black_box_func_gp_list[0].X))
            inputs[:, k] = [tuple(x) for x in np.array(agent_list[idx].black_box_func_gp_list[0].X)]
            idx += 1


    # Since MPI cannot directly handle arrays of tuples for numeric operations, convert it to a numerical form for MPI operations.
    # Example: Flattening the tuples if they contain numerical data.
    num_elements_per_tuple = config_instance.x_dim
    flat_inputs = np.zeros((len(agent_list[0].black_box_func_gp_list[0].X) * num_elements_per_tuple, num_agent))
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            if inputs[i, j] is not None:
                flat_inputs[i * num_elements_per_tuple:(i + 1) * num_elements_per_tuple, j] = inputs[i, j]

    total_flat_inputs = np.zeros_like(flat_inputs)
    comm.Allreduce(flat_inputs, total_flat_inputs, op=MPI.SUM)

    # Optionally convert flat numerical data back to tuples if necessary
    total_inputs = np.empty_like(inputs)
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            start_idx = i * num_elements_per_tuple
            total_inputs[i, j] = tuple(total_flat_inputs[start_idx:start_idx + num_elements_per_tuple, j])

    total_inputs = pd.DataFrame(total_inputs)

    return agent_list, total_regret.reshape(-1), constr_black_total, constr_affine_total.reshape(-1), num_agent, total_inputs, lambda_list, mu_list





def run_instances(instance_num, config_instance):
    
    # Initialization of data storage lists
    agent_lists = []
    regrets = []
    black_constraints = []
    affine_constraints = []
    lambda_lists = []

    # Running multiple instances
    # for _ in tqdm(range(instance_num)):
    for _ in range(instance_num):
        agents, regret, black_constr, affine_constr, num_agents, inputs, lambdas, _ = DMABO(config_instance)
        agent_lists.append(agents)
        regrets.append(regret)
        black_constraints.append(black_constr)
        affine_constraints.append(affine_constr)
        lambda_lists.append(lambdas)

    # Calculating means across all instances
    black_mean = np.mean(black_constraints, axis=0)
    affine_mean = np.mean(affine_constraints, axis=0)
    regret_mean = np.mean(regrets, axis=0)
    # lambda_mean = np.mean(lambda_lists, axis=0)

    num_agents = config_instance.num_agents
    num_black_box_constraints = agent_lists[0][0].num_black_box_constrs

    return agent_lists, inputs, regret_mean, black_mean, affine_mean, lambda_lists, num_agents, num_black_box_constraints





# ----------------------------------------------------------------------------------------
# NEW BOPTEST THINGS
# ----------------------------------------------------------------------------------------



# DISCOMFORT
# ----------------------


# PERIODS

def generate_working_period():
    # Determine if the person stays home all day
    if random.random() < 0.24:  # 24% chance of staying home
        # Person stays home all day, we can represent this with None or a specific indication
        return "Stays home all day", "Stays home all day"
    

    x_points = [0, 7.5, 8.5, 9.5, 10.5, 16.5, 17.5, 18.5, 19.5, 21.5, 22.5, 24]
    y_points = [1, 1, 0.88, 0.4, 0.24, 0.24, 0.3, 0.55, 0.9, 0.9, 1, 1]
    # Create a piecewise linear interpolation
    f = interp1d(x_points, y_points, kind='linear')

    # Generate leaving time between 7:30 AM (7.5) and 10:30 AM (10.5)
    leaving_time = get_weighted_random_time(7.5, 10.5, f, is_leaving=True)
    
    # Generate return time between 4:30 PM (16.5) and midnight (24)
    returning_time = get_weighted_random_time(16.5, 24, f, is_leaving=False)

    return leaving_time, returning_time

# Helper function to select weighted random times based on the interpolation function
def get_weighted_random_time(start, end, f, is_leaving=True):

    times = np.linspace(start, end, num=int((end-start) * 4 + 1))  # More points for better granularity
    f_values = f(times)
    
    if is_leaving:
        weights = -np.diff(f_values)  # Negative because lower f means higher chance of leaving
        weights = np.append(weights, weights[-1])  # Handle the last missing weight
    else:
        weights = np.diff(f_values, prepend=f_values[0])  # Higher f means higher chance of returning
    
    weights = np.maximum(weights, 0)  # Ensure all weights are non-negative
    weights += 1e-8  # Avoid zero probability
    
    selected_time = random.choices(times, weights=weights, k=1)[0]
    hours = int(selected_time)
    minutes = int((selected_time - hours) * 60)
    
    return f"{hours:02}:{minutes:02}"


def generate_sleeping_period():
    # Define the start and end of the allowed time range (22:00 to 08:00)
    start_minute = 22 * 60  # 10 PM in minutes
    end_minute = 32 * 60    # 8 AM next day in minutes

    # Randomly choose a start time for the 7-hour (420 minutes) sleeping period
    # We subtract 420 because we need to fit a 7-hour block starting from the chosen time
    sleep_time = random.choice(range(6 * 60, 8 * 60 + 15, 15))
    
    sleep_start = random.choice(range(start_minute, end_minute - sleep_time, 15))
    sleep_end = sleep_start + sleep_time
    
    # Convert minutes back into HH:MM format
    sleep_start_hours = (sleep_start // 60) % 24
    sleep_start_minutes = sleep_start % 60
    sleep_end_hours = (sleep_end // 60) % 24
    sleep_end_minutes = sleep_end % 60
    
    sleep_start_time = f"{sleep_start_hours:02}:{sleep_start_minutes:02}"
    sleep_end_time = f"{sleep_end_hours:02}:{sleep_end_minutes:02}"
    return sleep_start_time, sleep_end_time


def generate_day_schedule():
    # Get working and sleeping periods
    working_period = generate_working_period()
    sleeping_period = generate_sleeping_period()

    # Initialize status for each 15-minute block of the day
    # 0: at work, 1: sleeping, 2: at home
    day_status = [2] * 96  # 24 hours * 4 blocks per hour

    if working_period[0] != "Stays home all day":
        # Parse times and convert to 15-minute blocks
        leave_time = (int(working_period[0][:2]) * 60 + int(working_period[0][3:5])) // 15
        return_time = (int(working_period[1][:2]) * 60 + int(working_period[1][3:5])) // 15

        # Update status for working period
        for block in range(leave_time, return_time):
            day_status[block] = 0  # At work

    # Update status for sleeping period
    sleep_start = (int(sleeping_period[0][:2]) * 60 + int(sleeping_period[0][3:5])) // 15
    sleep_end = (int(sleeping_period[1][:2]) * 60 + int(sleeping_period[1][3:5])) // 15

    if sleep_end < sleep_start:  # Crosses midnight
        sleep_end += 96  # Total number of 15-minute blocks in a day

    for block in range(sleep_start, sleep_end):
        if block < 96:
            day_status[block] = 1  # Sleeping
        else:
            day_status[block - 96] = 1  # Sleeping after midnight

    return day_status

# DICOMFORT

def discomfort(indoor_temp, min_temp, max_temp, sleep_min, sleep_max, status):
    if status == 2: # At home
        if indoor_temp < min_temp:
            return (min_temp - indoor_temp)#**2
        elif indoor_temp > max_temp:
            return (indoor_temp - max_temp)#**2
        else:
            return 0
    elif status == 0: # Sleeping
        if indoor_temp < sleep_min:
            return (sleep_min - indoor_temp)#**2
        elif indoor_temp > sleep_max:
            return (indoor_temp - sleep_max)#**2
        else:
            return 0
    else:  # At work
        return 0


def base_temp_proflies_agent(agent_id):

    # DAY SCHEDULE
    day_profile = pd.Series(generate_day_schedule(), index=pd.date_range(start=0, periods=96, freq='15min'), name='daily_profile')
    day_profile.index = day_profile.index.time
    day_profile.index.name = 'time'
    if PLOT_PROFILE:
        plt.plot(day_profile.values)
        plt.show()

    # OUTDOOR TEMP
    out_temp = pd.read_csv('input_files/out_temp.csv', index_col=0)
    out_temp = out_temp['weaSta_reaWeaTDryBul_y']
    out_temp.index = pd.to_datetime(out_temp.index, unit='s', origin='unix')
    out_temp.name = 'temperature'

    # COMPUTE WEEKLY EFFECTIVE TEMP
    df_6am = out_temp.between_time('06:00', '06:00')
    df_3pm = out_temp.between_time('15:00', '15:00')

    df_daily = pd.DataFrame({
        'temp_6am': df_6am,
        'temp_3pm': df_3pm
    })

    df_daily['effective_temp'] = df_daily.mean(axis=1)
    mean_daily_eff_temp = df_daily['effective_temp'].resample('D').mean().copy()

    # # Calculate the 7-day rolling average of effective temperatures
    mean_weekly_eff_temp = mean_daily_eff_temp.rolling(window=7).mean()
    mean_weekly_eff_temp[(mean_weekly_eff_temp.index[-1] + pd.Timedelta(days=1)).strftime('%Y-%m-%d')] = 0

    # COMFORT TEMPERATURE
    comfort_temp = 22.6 + 0.04 * (mean_weekly_eff_temp)
    comfort_temp = comfort_temp.resample('15min').ffill()
    comfort_temp.drop(comfort_temp.tail(3).index,inplace=True)


    dev_80 = np.random.normal(2.4, 1.12)  # How wide the gap is
    while not (2.4 - 1.12*2 <= dev_80 <= 2.4 + 1.12*2): dev_80 = np.random.normal(2.4, 1.12)
    dev_mean = np.random.normal(0, 1.91)   # Deviation from the mean comfort temperature
    while not (-1.91 <= dev_mean <= 1.91): dev_mean = np.random.normal(0, 1.91)
    dev_mean = 0 # This makes sense according to the paper

    comfort_temp_80_min = comfort_temp + dev_mean - dev_80
    comfort_temp_80_max = comfort_temp + dev_mean + dev_80
    # if agent_id == rank:
    if True:
        print(f'rank: {rank} - comfort_temp_80_min', comfort_temp_80_min.min())

    comfort_temp.name = 'comfort_temp'
    comfort_temp_80_min.name = 'comfort_temp_80_min'
    comfort_temp_80_max.name = 'comfort_temp_80_max'

    if PLOT_PROFILE:
        plt.figure(figsize=(8, 4))
        plt.plot(comfort_temp + dev_mean, label='Comfort Temperature')
        plt.fill_between(comfort_temp.index, comfort_temp_80_min, comfort_temp_80_max, alpha=0.3, label='80% Comfort Range')
        plt.title('Comfort Temperature and 80% Comfort Range')
        plt.show()

    comfort_df = pd.concat([comfort_temp_80_min, comfort_temp_80_max], axis=1, join='inner')
    comfort_df['sleep_min'] = 17
    comfort_df['sleep_max'] = 26


    comfort_status = pd.DataFrame(index = comfort_temp.index)
    comfort_status.index.name = None
    comfort_status['time'] = comfort_status.index.time

    comfort_status = pd.merge(comfort_status, day_profile, on='time', how='left')
    comfort_status = pd.Series(comfort_status['daily_profile'])

    comfort_df['comfort_status'] = comfort_status.values
    
    return comfort_df


def calculate_discomfort(comfort_df, temp_profile):

    times_after_warmup = np.load('input_files/times_after_warmup.npy')

    indoor_temp = pd.Series(temp_profile, index=pd.date_range(start=pd.to_datetime(times_after_warmup[0], unit='s', origin='unix'), 
                                                              periods=len(temp_profile), freq='h'), name='indoor_temp')

    indoor_temp = indoor_temp.resample('15min').ffill()
    indoor_temp = indoor_temp

    if PLOT_PROFILE:
        plt.plot(indoor_temp)
        plt.show()

    comfort_df = pd.concat([comfort_df, indoor_temp], axis=1, join='inner')

    comfort_df['discomfort'] = comfort_df.apply(lambda x: discomfort(x['indoor_temp'], 
                                                                 x['comfort_temp_80_min'], x['comfort_temp_80_max'], 
                                                                 x['sleep_min'], x['sleep_max'],
                                                                 x['comfort_status']), axis=1)
    
    return comfort_df['discomfort'].sum() / 4


# ELECTRICITY CONSUMPTION
# ----------------------

def calculate_welec(welec):
    sums = np.zeros(N_PERIODS)
        
    for hour in range(len(welec)):
        period = categorize_time(hour)
        sums[period] += welec[hour]
    
    return sums
