# Distributed Multi-Agent Bayesian Optimization in Building Temperature Control.

This work demonstrates the applicability of a novel Distributed Multi-Agent Bayesian Optimization (DMABO) algorithm in managing temperature control for smart buildings via black-box learning. The primary focus lies on optimizing HVAC systems across a network of residential buildings, utilizing demand response programs to reduce peak load demands. Despite the growing interest in Bayesian Optimization, there remains a notable gap in research concerning its integration within multi-agent systems for demand response applications. This methodology employs Bayesian Optimization to address the complex, derivative-free optimization challenges posed by expensive black-box functions. The algorithm integrates a multi-agent setup that handles individual building controls while collectively aiming to improve overall energy efficiency and thermal comfort using a primal-dual formulation. Simulations demonstrate that this approach effectively reduces peak energy loads and maintains desired indoor temperatures, outperforming single agent optimizations. Additionally, this study establishes a framework for building comfort and scheduling, details the parallelization of the algorithm using the Message Passing Interface (MPI), and outlines both current limitations and directions for future research.


## Getting Started

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/carlos-collado/bayesian-opt-multi-agent.git
   cd bayesian-opt-multi-agent
   ```

2. **Create and Activate the Conda Environment**:

   ```bash
   conda env create -f environment.yml
   conda activate bayesian-opt
   ```

## Running the Optimization

To run the distributed multi-agent Bayesian optimization:

```bash
python dmabo_run.py
```

This code runs the optimization and creates the following plots:

This code runs the optimization and creates the following plots:

1. **Convergence Plot**: Shows the objective function value over iterations, demonstrating how the optimization converges toward the optimal solution.

2. **Agent Temperature Profiles**: Visualizes the temperature trajectories for each building agent, showing how indoor temperatures are maintained within comfort constraints.

3. **Power Consumption Profile**: Displays the power consumption over time for all agents, highlighting peak load reduction compared to baseline scenarios.

4. **Constraint Violation**: Tracks constraint violations throughout the optimization process, showing how the algorithm manages to satisfy both local and global constraints.

5. **Dual Variables Evolution**: Shows how the dual variables (Lagrange multipliers) evolve during the optimization, representing the "price signals" that coordinate the agents.


## Understanding the Optimization

The DMABO algorithm is implemented through several key components:

1. **agent_config.py**:  
   - Specifies configuration details for each building agent (e.g., comfort range, maximum power, or any agent-specific parameters).

2. **bo_agent.py**:  
   - Implements the `BOAgent` class, defining how each agent performs Bayesian Optimization updates on its local objective while communicating with the coordinator for global constraints.

3. **coordinator.py**:  
   - Acts as the central communication point. It collects information from each agent, updates and broadcasts the dual variables (price signals), and ensures that constraints are satisfied across all agents.

4. **dmabo_instance.py**:  
   - Sets up and runs the multi-agent simulation. It instantiates the agents, initializes the coordinator, and orchestrates the iterative optimization cycle.

5. **functions_to_sample.py**:  
   - Contains the underlying objective functions or cost functions each agent aims to optimize, including any power/temperature models used by the building simulations.

6. **utils.py**:  
   - Provides utility functions (e.g., data processing, plotting helpers, or logging) to support the main optimization loop.

When you run `dmabo_run.py`, it combines all these components, handles the initialization of agents and coordinator, orchestrates the communication and updates in each iteration, and creates the plots described above. This design allows you to easily customize agent behavior, define new objective functions, or adjust the constraints to suit different building configurations and energy management scenarios.



























