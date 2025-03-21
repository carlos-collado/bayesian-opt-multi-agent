import numpy as np
import GPy
import safeopt
from util import objective, black_box_1, stratified_uniform_sample_lhs, boptest_legthscale, boptest_variance
from tqdm import tqdm

BIG_M = 1e10


def get_agent_config(agent_id, config_instance, objective_only=False):
    
    config = dict()
    config['agent_id'] = agent_id


    # GLOBAL PROBLEM CONFIGURATION

    config['problem_name'] = config_instance.problem_name
    config['num_agents'] = config_instance.num_agents
    config['eta'] = config_instance.eta
    config['beta'] = config_instance.beta
    config['run_horizon'] = config_instance.run_horizon

    num_samples = config_instance.num_samples
    discrete_num_per_dim = config_instance.discrete_num_per_dim
    black_box_funcs_dim = config_instance.black_box_funcs_dim



    # INPUT X
    x_dim = config_instance.x_dim
    x_range = config_instance.x_range

    x_grid = safeopt.linearly_spaced_combinations([x_range] * x_dim, [discrete_num_per_dim] * x_dim)

    config['x_grid'] = x_grid


    # BLACK-BOX FUNCTIONS (obj + constrs)
    black_box_funcs_list = []
    GP_kernel_list = []

    # Generate sample data
    X_sample = stratified_uniform_sample_lhs(x_range, x_dim, num_samples)
    Y_samples = []

    noise_f_sample = config_instance.noise_f_sample
    fcn = config_instance.fcn

    for k in tqdm(range(black_box_funcs_dim), desc=f"Agent {agent_id} - Sample data"):

        # --- Black-box functions ---
        if k == 0:
            func = objective(fcn = fcn, noise = noise_f_sample, agent_id = agent_id)
            if objective_only:
                break
        else:
            func = black_box_1(fcn = fcn, noise = noise_f_sample, agent_id = agent_id, num_bb = k)

        black_box_funcs_list.append(func)
        Y_samples.append(np.array([func(x) for x in X_sample]).reshape(-1, 1))

        # --- GP and beta ---
        variance_estimate, min_variance, max_variance = boptest_variance(k, fcn, normalize=True, bounds=True)
        lengthscale_estimate, min_lengthscale, max_lengthscale = boptest_legthscale(x_range, normalize=True)

        kernel = GPy.kern.RBF(input_dim=x_dim, variance=variance_estimate, lengthscale=lengthscale_estimate, ARD=True)

        # Kernel constraint
        kernel.lengthscale.constrain_bounded(min_lengthscale, max_lengthscale, warning=False)
        kernel.variance.constrain_bounded(min_variance, max_variance, warning=False)

        GP_kernel_list.append(kernel)

    config['fcn'] = fcn
    config['num_black_box_funcs'] = black_box_funcs_dim
    config['num_black_box_constrs'] = black_box_funcs_dim - 1
    config['black_box_funcs_list'] = black_box_funcs_list
    config['kernel_list'] = GP_kernel_list
    config['X_sample'] = X_sample
    config['Y_samples'] = Y_samples
    config['x_range'] = x_range

    # OTHER PARAMETERS
    config['local_A'] = np.ones(1) # 1 because we don't do any operation on input to calculate affine constr.

    return config