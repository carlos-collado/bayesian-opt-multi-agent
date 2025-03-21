"""
Implement the agent coordinator.
"""
import numpy as np
# from memory_profiler import profile
from util import cosine_annealing_lr
from mpi4py import MPI
import time

class Coordinator():

    def __init__(self, coordinator_config):

        # COPY PARAMETERS FROM COORDINATOR CONFIG

        self.coordinator_config = coordinator_config
        self.num_agents = coordinator_config['num_agents']
        self.agent_list = coordinator_config['agent_list']
        self.num_blackbox_constraints = coordinator_config['num_blackbox_constraints']
        self.num_affine_constraints = coordinator_config['num_affine_constraints']

        self.epsilon = coordinator_config['epsilon']
        self.max_affine = coordinator_config['max_affine']
        self.max_blackbox = coordinator_config['max_blackbox']
        self.eta = coordinator_config['eta']
        self.run_horizon = coordinator_config['run_horizon']
        self.lr_function = cosine_annealing_lr(1, self.run_horizon)

        # Initialize dual variables
        self.lambda_dual = np.ones(self.num_blackbox_constraints, dtype=float) * 0
        self.mu_dual = np.zeros(self.num_affine_constraints)
        self.indices_to_update = None

        # MPI

        self.comm = MPI.COMM_WORLD  # Get the communicator object
        self.rank = self.comm.Get_rank()  # Get the rank of the current process
        self.size = self.comm.Get_size()  # Get the total number of processes

        if self.rank == 0:
            print(f"Total number of processes: {self.size}")
            print(f"Rank of the current process: {self.rank}")

        # Example for equally distributing agents across MPI processes
        agents_per_process = len(self.agent_list) // self.size
        start_index = self.rank * agents_per_process
        end_index = start_index + agents_per_process if self.rank != self.size - 1 else len(self.agent_list)

        self.local_agents = self.agent_list[start_index:end_index]
        self.start_index = start_index
        self.end_index = end_index
        
        self.time_primal = 0
        self.time_dual = 0



    '''
    def primal_update(self, t):
        local_Ax_sum = 0
        local_constr_lg_sum = 0

        for bo_agent in self.agent_list:
            local_Ax, lowerg = bo_agent.local_pd_primal_update(self.lambda_dual, self.mu_dual, t, self.indices_to_update)

            # sum over all agents
            local_Ax_sum += local_Ax
            local_constr_lg_sum += lowerg

        return local_Ax_sum, local_constr_lg_sum
    '''


    def primal_update(self, t):
        local_Ax_sum = 0
        local_constr_lg_sum = 0

        for bo_agent in self.local_agents:  # local_agents is the subset of agents
            local_Ax, lowerg = bo_agent.local_pd_primal_update(self.lambda_dual, self.mu_dual, t, self.indices_to_update, self.max_blackbox)
            local_Ax_sum += local_Ax
            local_constr_lg_sum += lowerg

        # Aggregate results from all processes
        total_Ax_sum = self.comm.allreduce(local_Ax_sum, op=MPI.SUM)
        total_constr_lg_sum = self.comm.allreduce(local_constr_lg_sum, op=MPI.SUM)

        return total_Ax_sum, total_constr_lg_sum






    def dual_update(self, local_Ax_sum, local_constr_lg_sum, t):

        bb_violation = ((local_constr_lg_sum + self.epsilon) - self.max_blackbox) # * (1 + 0.5 * self.lr_function(t)) # start with relaxed constraints
        # bb_violation = np.where(bb_violation > 0, bb_violation**2, bb_violation)
        lambda_update = np.maximum(self.lambda_dual + bb_violation, 0)

        # alpha = np.where(bb_violation > 0, 0.2, 0.1)
        alpha = self.lr_function(t) * 0.2

        # if t > 200:
        # alpha = 1e-6

        # if t % 5 == 0:

        self.indices_to_update = self.select_indices_to_update(bb_violation)
        self.indices_to_update = None
        if self.indices_to_update == None:
            self.lambda_dual = (1 - alpha) * self.lambda_dual + alpha * lambda_update
        else:
            for idx in self.indices_to_update:
                self.lambda_dual[idx] = (1 - alpha) * self.lambda_dual[idx] + alpha * lambda_update[idx]

        # self.lambda_dual = (1 - alpha) * self.lambda_dual + alpha * lambda_update
        # self.lambda_dual[1] = 10000

        affine_constraint_violation = local_Ax_sum - self.max_affine
        if affine_constraint_violation > 0:
            raise ValueError('Affine constraint violation')
            self.mu_dual += affine_constraint_violation
        self.mu_dual = np.maximum(self.mu_dual, 0)


    def select_indices_to_update(self, bb_violation):
        valid_indices = np.where(bb_violation >= 0)[0] # indices with non-negative violations
        valid_indices = np.array([])
        
        if valid_indices.size > 0:
            num_updates = 1
            indices = valid_indices[np.argsort(bb_violation[valid_indices])[-num_updates:]]
        else:
            indices = np.arange(len(bb_violation))
        # indices = np.random.choice(len(bb_violation), num_updates, replace=False) # random indices

        return indices

    def update(self, t):
        time_start = time.time()

        # PRIMARY UPDATE
        local_Ax_sum, local_constr_lg_sum = self.primal_update(t)
        time_primal = time.time()
        self.time_primal += time_primal - time_start

        # DUAL UPDATE
        if self.rank == 0:
            self.dual_update(local_Ax_sum, local_constr_lg_sum, t)
        # Broadcast updated dual variables from root to all other processes
        self.lambda_dual = self.comm.bcast(self.lambda_dual, root=0)
        self.mu_dual = self.comm.bcast(self.mu_dual, root=0)
        self.time_dual += time.time() - time_primal 
