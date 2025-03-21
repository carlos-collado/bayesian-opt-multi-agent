# -*- coding: utf-8 -*-
"""
This module implements a simple P controller of heater power control specifically
for testcase1.

"""


def compute_control(y, Kp, LowerSetp = 273.15+20, UpperSetp = 273.15+23):
    """Compute the control input from the measurement.

    Parameters
    ----------
    y : dict
        Contains the current values of the measurements.
        {<measurement_name>:<measurement_value>}
    forecasts : structure depends on controller, optional
        Forecasts used to calculate control.
        Default is None.

    Returns
    -------
    u : dict
        Defines the control input to be used for the next step.
        {<input_name> : <input_value>}

    """    
    # check if Kp is a list
    if isinstance(Kp, list):
        k_p = Kp[0]
    else:
        k_p = Kp
    
    # Compute control
    if y['reaTZon_y'] < LowerSetp:
        e = LowerSetp - y['reaTZon_y']
    elif y['reaTZon_y'] > UpperSetp:
        e = UpperSetp - y['reaTZon_y']
    else:
        e = 0

    value = k_p*e
    # value = min(value, alpha)
    u = {
        'oveHeaPumY_u': value,
        'oveHeaPumY_activate': 1
    }

    return u


def initialize():
    """Initialize the control input u.

    Parameters
    ----------
    None

    Returns
    -------
    u : dict
        Defines the control input to be used for the next step.
        {<input_name> : <input_value>}

    """

    u = {
        'oveHeaPumY_u': 0,
        'oveHeaPumY_activate': 1
    }

    return u
