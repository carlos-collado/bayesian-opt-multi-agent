"""
Implement a BO agent.
"""
import numpy as np
import GPy
import matplotlib.pyplot as plt
from util import transform_to_2d, boptest_noise_variance, boptest_norm, stratified_uniform_sample_lhs
from tabulate import tabulate
import time
from tqdm import tqdm
from scipy.optimize import minimize, minimize_scalar

class BO_AGENT():

    def __init__(self, bo_agent_config):

        # COPY PARAMETERS FROM AGENT CONFIG

        self.agent_id = bo_agent_config['agent_id']
        self.num_agents = bo_agent_config['num_agents']

        self.bo_agent_config = bo_agent_config
        self.black_box_funcs_list = bo_agent_config['black_box_funcs_list']
        self.num_black_box_funcs = bo_agent_config['num_black_box_funcs']
        self.num_black_box_constrs = bo_agent_config['num_black_box_constrs']
        self.x_grid = bo_agent_config['x_grid']
        self.x_range = bo_agent_config['x_range']

        num_x, dim_x = bo_agent_config['x_grid'].shape

        self.num_x = num_x
        self.dim_x = dim_x

        self.X_eval = bo_agent_config['X_sample']
        self.Y_eval = bo_agent_config['Y_samples']
        self.eta = bo_agent_config['eta']
        self.local_A = bo_agent_config['local_A']
        self.fcn = bo_agent_config['fcn']
        beta0_init, beta1_init = bo_agent_config['beta']
        # self.beta0 = cosine_annealing_lr(beta0_init, bo_agent_config['run_horizon'])
        # self.beta1 = cosine_annealing_lr(beta1_init, bo_agent_config['run_horizon'])
        self.beta0 = beta0_init
        self.beta1 = beta1_init

        self.run_horizon = bo_agent_config['run_horizon']

        self.obj_lcb = []
        self.constrain_lcb = []

        self.x0 = None


        # BB FUNCTIONS GAUSSIAN PROCESS

        black_box_func_gp_list = []

        for k in tqdm(range(self.num_black_box_funcs), desc=f"Agent {self.agent_id} - GP fit"):

            noise_variance, min_noise_variance, max_noise_variance = boptest_noise_variance(k, self.fcn, normalize=True)

            model = GPy.models.GPRegression(self.X_eval, transform_to_2d(self.Y_eval[k]),
                                                                    bo_agent_config['kernel_list'][k], 
                                                                    noise_var=noise_variance,)

            # Model constraint
            model.Gaussian_noise.variance.constrain_bounded(min_noise_variance, max_noise_variance, warning=False)

            # Optimize model
            model.optimize_restarts(num_restarts=10, verbose=False)
            black_box_func_gp_list.append(model)

        self.black_box_func_gp_list = black_box_func_gp_list



    def local_pd_primal_update(self, lambda_dual, mu_dual, t, var_dim, max_blackbox):

        local_max_bb = max_blackbox / self.num_agents * 1.2
        
        gp_obj = self.black_box_func_gp_list[0]
        gp_constr_list = self.black_box_func_gp_list[1:]
        self.gp_obj = gp_obj
        self.gp_constr_list = gp_constr_list

        Cij = [1.25 * boptest_norm(k, self.fcn, normalize=True) for k in range(self.num_black_box_funcs)]
        beta0, beta1, eta = self.beta0, self.beta1, self.eta
        self.betas = [beta0, beta1]

        bounds = [(self.x_range[0], self.x_range[1])]
        if var_dim is None:
            bounds = bounds * self.dim_x
        elif type(var_dim) != int:
            bounds = bounds * len(var_dim)

        # print(f'it {t}  var_dim: {var_dim} - lambda_dual: {lambda_dual}')


        def predict_mean_and_variance(gp, x):
            mean, var = gp.predict(np.array([x]))
            return mean[0], var[0]


        def objective_function(x0=None, var_dim=None, return_lcb_constraints=False):
            def objective(x):
                if x0 is None:
                    x_to_opt = x
                else:
                    if var_dim is None:
                        x_to_opt = x
                    elif (type(var_dim) == int):
                        x_to_opt = np.array(x0, copy=True)
                        x_to_opt[var_dim] = x
                    elif len(var_dim) == 1:
                        x_to_opt = np.array(x0, copy=True)
                        x_to_opt[var_dim[0]] = x
                    else:
                        x_to_opt = np.array(x0, copy=True)
                        for i, var in enumerate(var_dim):
                            x_to_opt[var] = x[i]

                # OBJECTIVE
                mean, var = predict_mean_and_variance(gp_obj, x_to_opt)
                penalty = beta0 * np.sqrt(var)
                lcb_obj = mean - penalty

                # CONSTRAINTS
                lcb_constr = []
                for i, gp in enumerate(gp_constr_list):
                    mean_c, var_c = predict_mean_and_variance(gp, x_to_opt)
                    penalty_c = beta1 * np.sqrt(var_c)
                    lcb_constr.append(np.maximum(mean_c - penalty_c, -Cij[i+1]).item())

                if return_lcb_constraints:
                    return np.array(lcb_constr)

                aux_obj = lcb_obj + eta * np.sum(lcb_constr * lambda_dual)

                return aux_obj.item()
        
            return objective



        '''
        def optimize_with_retries(objective, x_range, bounds, num_starts=50, max_retries=4):
            retry_count = 0
            best_result = None

            is_one_dim = len(bounds) == 1 and bounds[0][0] != None and bounds[0][1] != None

            while retry_count <= max_retries:

                if is_one_dim:
                    starting_points = np.linspace(bounds[0][0], bounds[0][1], num_starts)
                else:
                    starting_points = stratified_uniform_sample_lhs(x_range, len(bounds), num_starts)

                for x0 in starting_points:
                    if is_one_dim:
                        if type(bounds) == list:
                            bounds = bounds[0]
                        result = minimize_scalar(
                            objective, bounds=bounds, method='bounded'
                        )
                    else:
                        result = minimize(
                            objective, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxfun': 50000, 'maxiter': 50000, 'ftol': 1e-06, 'gtol': 1e-05}
                        )

                    if best_result is None or result.fun < best_result.fun:
                        best_result = result

                # Check if a successful optimization has been found
                if best_result is not None and best_result.success:
                    break

                retry_count += 1
                num_starts *= 2  # Increase the number of starting points for the next retry if needed
                print(f"Retry {retry_count}: Increasing number of starting points to {num_starts}")

            if best_result is None or not best_result.success:
                print("Failed to find a successful optimization result after retries.")

            return best_result

        result = optimize_with_retries(objective_function(self.x0, var_dim), self.x_range, bounds)

        '''




        def optimize_with_retries(objective_func, constraints_func, x_range, bounds, num_starts=50, max_retries=4):
            retry_count = 0
            best_result = None

            is_one_dim = len(bounds) == 1 and bounds[0][0] is not None and bounds[0][1] is not None

            # Define constraints in the format required by scipy.optimize.minimize
            num_constraints = len(local_max_bb)
            constraint_list = []
            for i in range(num_constraints):
                def constraint_factory(j):
                    return {'type': 'ineq', 'fun': lambda x, j=j: local_max_bb[j] - constraints_func(x)[j]}
                constraint_list.append(constraint_factory(i))

            while retry_count <= max_retries:

                if is_one_dim:
                    starting_points = np.linspace(bounds[0][0], bounds[0][1], num_starts).reshape(-1, 1)
                else:
                    starting_points = stratified_uniform_sample_lhs(x_range, len(bounds), num_starts)

                for x0 in starting_points:
                    if is_one_dim:
                        x0 = x0[0]

                    result = minimize(
                        objective_func,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraint_list,
                        options={'maxiter': 50000, 'ftol': 1e-06}
                    )

                    if best_result is None or result.fun < best_result.fun:
                        best_result = result

                if best_result is not None and best_result.success:
                    break

                retry_count += 1
                num_starts *= 2
                print(f"Retry {retry_count}: Increasing number of starting points to {num_starts}")

            if best_result is None or not best_result.success:
                print("Failed to find a successful optimization result after retries.")

            return best_result

        objective_func = objective_function(self.x0, var_dim, return_lcb_constraints=False)
        constraints_func = objective_function(self.x0, var_dim, return_lcb_constraints=True)

        # Perform optimization with constraints
        result = optimize_with_retries(objective_func, constraints_func, self.x_range, bounds)





        if self.x0 is None:
            opt_x = result.x
        else:
            if var_dim is None:
                opt_x = result.x
            elif type(var_dim) == int:
                opt_x = np.array(self.x0, copy=True)
                opt_x[var_dim] = result.x
            elif len(var_dim) == 1:
                opt_x = np.array(self.x0, copy=True)
                opt_x[var_dim[0]] = result.x
            else:
                opt_x = np.array(self.x0, copy=True)
                for i, var in enumerate(var_dim):
                    opt_x[var] = result.x[i]


        self.x0 = opt_x

        # Compare with grid search
        if False:
            obj_fun = [objective_function(x) for x in self.x_grid]
            grid_opt_x = self.x_grid[np.argmin(obj_fun)]
            
            if not np.all(grid_opt_x == opt_x):
                print(f'Optimization failed. grid_opt_x = {grid_opt_x}, opt_x = {opt_x}')

        # Evaluate optimal (aux) in real function
        eval_result = self.local_evaluate(opt_x)
        self.X_eval = np.append(self.X_eval, np.array([opt_x]), axis=0)

        # Update GP
        self.update_local_gp(opt_x, eval_result)

        # local_Ax = self.local_A @ opt_x
        local_Ax = 0


        # Return lcb of constraints
        if self.num_black_box_constrs > 0:
            lowerg = objective_function(return_lcb_constraints=True)(opt_x)
        else:
            lowerg = 0

        self.constrain_lcb.append(lowerg)
        self.objective_function = objective_function(return_lcb_constraints=False)

        return local_Ax, lowerg


    def local_evaluate(self, x):
        x_copy = np.array(x, copy=True)
        eval_result = []
        for k in range(self.num_black_box_funcs):
            eval_result.append(self.black_box_funcs_list[k](x_copy))

        return eval_result


    def update_local_gp(self, x, eval_result):
        for k in range(self.num_black_box_funcs):
            y = eval_result[k]
            gp = self.black_box_func_gp_list[k]
            new_gp = self.update_single_gp(gp, x, y)
            new_gp.optimize(messages=False)
            # new_gp.optimize_restarts(num_restarts=10, verbose=False)
            self.black_box_func_gp_list[k] = new_gp
            self.Y_eval[k] = np.append(self.Y_eval[k], np.array([[y]]), axis=0)


    def update_single_gp(self, gp, x, y):
        # Add this to the GP model
        prev_X = gp.X
        prev_Y = gp.Y

        new_X = np.vstack([prev_X, x])
        new_obj = np.vstack([prev_Y, y])
        gp.set_XY(new_X, new_obj)
        return gp
