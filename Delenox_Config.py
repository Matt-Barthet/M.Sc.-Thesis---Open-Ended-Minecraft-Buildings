import numpy as np
import os
import tensorflow as tf
import neat
import pickle
import bz2
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from tensorflow.python.keras.utils.np_utils import to_categorical
from scipy.ndimage import center_of_mass
from scipy.spatial import distance
import time


def sinc(x):
    x = max(-60.0, min(60.0, 5.0 * x))
    return (np.sin(x) + 1) / 2


# General Parameters
thread_count = 12
number_of_phases = 10
target_species_count = 10

# Parameters for input space of un/compressed buildings
lattice_dimensions = (20, 20, 20)
activations = np.linspace(0, 1, lattice_dimensions[0])
value_range = [(x, y, z) for x in range(lattice_dimensions[0]) for y in range(lattice_dimensions[0]) for z in
               range(lattice_dimensions[0])]

# Auto-Encoder parameters for architecture and learning
batch_size = 64
no_epochs = 100
compressed_length = 256
loss_function = "categorical_crossentropy"
accuracy_metrics = ['categorical_accuracy', 'binary_accuracy']

# NEAT parameters for building generation and evolution
runs_per_phase = 10
population_size = 200
best_fit_count = int(1000 / runs_per_phase)
generations_per_run = 100
current_run = 0

# Parameters for constrained novelty search in the NEAT module
k_nearest_neighbors = 20
add_to_archive = 2

# Parameters for evolutionary algorithm using latent vector space
latent_mutation_rate = 0.1
latent_variable_range = [-250, 250]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

materials = {'External_Space': 0, 'Interior_Space': 1, 'Wall': 2, 'Floor': 3, 'Roof': 4}