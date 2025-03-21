import numpy as np
import pickle
import pandas as pd
import requests
import sys
from controllers.controller import Controller
from tqdm import tqdm


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

    def categorize_time(hour):
        if 21 <= hour or hour < 8:
            return 0  # Off-peak hours from 9:00 PM to 8:00 AM
        elif 16 <= hour < 21:
            return 1  # Peak hours from 4:00 PM to 9:00 PM
        else:
            return 2  # Normal hours from 8:00 AM to 4:00 PM

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
    welec_hour = w_elec.groupby(w_elec.index).sum()  / 1000 * 30 / 3600
    welec_hour = welec_hour.values

    return temp_profile, welec_hour




def simulation_output(inputs):
    tuple_inputs = tuple(inputs)
    with open('simulation_results.pickle', 'rb') as file:
        results = pickle.load(file)

    if tuple_inputs not in results.keys():
        print('Simulation not found: ', tuple_inputs)

        temp_profile, welec_hour = control_test(control_module='controllers.pid', length=40*24*3600, step=3600, Kp=1, 
                                        LowerSetp_list=tuple(inputs+273.15), warmup_period=0*24*3600, start_time=0*24*3600,
                                    warmup_days = 10)

        results[tuple_inputs] = {'welec': welec_hour, 'temp': temp_profile}

        with open('simulation_results.pickle', 'wb') as f:
            pickle.dump(results, f)
            print('Simulation saved: ', tuple_inputs)

    else:

        temp_profile = results[tuple_inputs]['temp']
        welec_hour = results[tuple_inputs]['welec']

    return temp_profile, welec_hour


# def simulation_output(inputs):
#     tuple_inputs = tuple(inputs)
#     results = {}

#     temp_profile, welec_hour = control_test(control_module='controllers.pid', length=40*24*3600, step=3600, Kp=1, 
#                                     LowerSetp_list=tuple(inputs+273.15), warmup_period=0*24*3600, start_time=0*24*3600,
#                                 warmup_days = 10)

#     results[tuple_inputs] = {'welec': welec_hour, 'temp': temp_profile}

#     with open('simulation_results_2_periods.pickle', 'wb') as f:
#         pickle.dump(results, f)
#         print('Simulation saved: ', tuple_inputs)

#     return temp_profile, welec_hour




input_combinations = [np.array([i, j, k]) for i in range(15, 25) for j in range(15, 25) for k in range(15, 25)]

for input in tqdm(input_combinations):
    simulation_output(input)