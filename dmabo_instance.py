"""
Implement the agent coordinator.
"""
from util import create_gifs, clear_directory, boptest_variance, comm, rank, size
from bo_agent import BO_AGENT
from agent_config import get_agent_config
from coordinator import Coordinator
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
from mpi4py import MPI

class PowerAllocationInstance():

    def __init__(self, config_instance):
        self.config_instance = config_instance
        self.num_agents = self.config_instance.num_agents
        self.run_horizon = self.config_instance.run_horizon


    def generate_one_instance(self):

        # BASICALLY COORDINATOR CONFIGURATION
        coordinator_config = dict()

        # Example for equally distributing agents across MPI processes
        agents_per_process = self.num_agents // size
        start_index = rank * agents_per_process
        end_index = start_index + agents_per_process if rank != size - 1 else self.num_agents


        # CREATE AGENTS
        coordinator_config['num_agents'] = self.num_agents
        agent_list = []
        for i in range(self.num_agents):
            if i >= start_index and i < end_index:
                agent_list.append( BO_AGENT(get_agent_config(i, self.config_instance)) )
            else:
                agent_list.append(None)
                # placeholder = get_agent_config(i, self.config_instance, objective_only=True)

        coordinator_config['agent_list'] = agent_list

        comm.Barrier()



        # CONSTRAINTS
        max_affine = self.config_instance.max_affine
        max_blackbox = self.config_instance.max_blackbox

        coordinator_config['num_blackbox_constraints'] = agent_list[start_index].num_black_box_constrs
        coordinator_config['num_affine_constraints'] = 1
        coordinator_config['max_affine'] = max_affine
        coordinator_config['max_blackbox'] = max_blackbox / np.sqrt(boptest_variance(4, normalize=False))
        if rank == 0:
            print(f'max_blackbox: {coordinator_config["max_blackbox"]}')
        coordinator_config['epsilon'] = self.config_instance.epsilon
        coordinator_config['eta'] = self.config_instance.eta
        coordinator_config['run_horizon'] = self.run_horizon


        return Coordinator(coordinator_config)




    def run_one_instance(self):

        start_time = time.time()
        pd_pac = self.generate_one_instance()
        gen_inst_time = time.time() - start_time
        # raise KeyError('This is a test error')
        lambda_list = []
        mu_list = [0]
        # time_list = []
        
        clear_directory('fig/GP_updates/png')
        plot_gif = self.config_instance.plot_gif


        if rank == 0:
            loop = tqdm(range(self.run_horizon))
        else:
            loop = range(self.run_horizon)

        for it in loop:
        # for it in range(self.run_horizon):
            # print(f'it {it} -- lambda: {pd_pac.lambda_dual}')
            # start_time = time.time()
            pd_pac.update(it) # to avoid zero elements

            lambda_list.append(pd_pac.lambda_dual.copy())
            mu_list.append(pd_pac.mu_dual[0])

            if plot_gif:

                fig, axes = plt.subplots(nrows=self.num_agents, ncols=pd_pac.agent_list[0].num_black_box_funcs, figsize=(15, 10))
                axes = np.array(axes).reshape(self.num_agents, -1)  # Ensure axes is always 2D even if one row

                for i in range(self.num_agents):
                    for j in range(pd_pac.agent_list[i].num_black_box_funcs):
                        # Determine the appropriate subplot
                        ax = axes[i, j] if self.num_agents > 1 else axes[j]

                        # Generate data and plot
                        x = np.linspace(0, 5, 100)
                        gp_obj = pd_pac.agent_list[i].black_box_func_gp_list[j]
                        opt_x = pd_pac.agent_list[i].opt_x
                        m, v = gp_obj.predict(x[:, np.newaxis])
                        m, v = m.flatten(), v.flatten()
                        ax.plot(x, m)
                        ax.scatter(pd_pac.agent_list[i].X_eval, pd_pac.agent_list[i].Y_eval[j], color='red', s=20)
                        ax.scatter(opt_x, gp_obj.predict(opt_x[:, np.newaxis])[0], color='green', s=50)
                        ax.fill_between(x, m - 2 * np.sqrt(v), m + 2 * np.sqrt(v), alpha=0.25)

                        ax.set_xlim(self.config_instance.x_range[0], self.config_instance.x_range[1])

                        # Add custom label for beta values
                        beta_info = f'  x = {(opt_x.item()):.3f}\n β0 = {pd_pac.agent_list[i].betas[0]:.3g}\n β1 = {pd_pac.agent_list[i].betas[1]:.3g}'
                        handles, _ = ax.get_legend_handles_labels()
                        custom_label = plt.Line2D([], [], color='none', label=beta_info)
                        handles.append(custom_label)
                        ax.legend(handles=handles, loc='upper left', fontsize='small')

                        ax.set_title(f'Agent {pd_pac.agent_list[i].agent_id}, Func {j} - it {it}')

                # Adjust layout to prevent overlap
                plt.tight_layout()
                fig.savefig(f'fig/GP_updates/png/gp_{it}.png')
                plt.close(fig)

            # end_time = time.time()
            # time_list.append(end_time - start_time)

        if plot_gif:
            create_gifs(self.num_agents, pd_pac.agent_list[0].num_black_box_funcs, self.run_horizon)

        if rank == 0:
            print(f'Generation time: {gen_inst_time}')
            print(f'Primal Time: {pd_pac.time_primal}')
            print(f'Dual Time: {pd_pac.time_dual}')
            # print(f'Time list: {time_list}')

        return pd_pac, lambda_list, mu_list