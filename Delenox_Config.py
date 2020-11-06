import numpy as np

# System Parameters
thread_count = 11

# Experiment Parameters
averaged_runs = 1

# Parameters for input space of un/compressed buildings
lattice_dimensions = (20, 20, 20)
activations = np.linspace(0, 1, lattice_dimensions[0])
value_range = [(x, y, z) for x in range(lattice_dimensions[0]) for y in range(lattice_dimensions[0]) for z in range(lattice_dimensions[0])]

# Auto-Encoder parameters for architecture and learning
batch_size = 128
no_epochs = 30
validation_split = 0.2

# NEAT parameters for initial building population generation
initial_runs = 1
best_fit_count = 100

# Parameters for evolutionary algorithm using latent vector space
latent_generations = 1
latent_mutation_rate = 0.1
latent_variable_range = [-250, 250]
