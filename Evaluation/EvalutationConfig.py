import os
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle

from scipy.stats import entropy
from scipy.special import softmax
from conda import iteritems
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA

from Generator.Autoencoder import load_model
from Generator.Constraints import *
from Generator.Autoencoder import convert_to_integer, calculate_error, add_noise
from Generator.Visualization import get_color_map, expressive_graph
from Generator.NeatGenerator import *
from Generator.Delenox_Config import *
import Generator.Delenox_Config


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
flatten = itertools.chain.from_iterable
plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('axes', labelsize=12)

config = load_config_file()
locations = [(0, 0), (0, 1), (1, 0), (1, 1)]
pca_locs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
novelty_spectrum_subplots = [[(5, 5, index * 5 + offset) for index in range(5)] for offset in range(1, 6)]
ae_label = ['Vanilla AE', 'Denoising AE']

phases_to_evaluate = 10

# labels = ["Random AE"]
labels = ["Static AE", "Random AE", "LS-AE", "FH-AE", "NA-AE"]
colors = ['black', '#d63333', '#3359d6', '#3398d6', '#662dc2']
markers = ['s', 'v', 'D', '^', 'o']
linestyles = ['solid', 'dashed', 'dashed', 'dashed', 'dashed']

neat_keys = ["Node Complexity", "Connection Complexity", "Archive Size", "Best Novelty", "Mean Novelty",
             "Infeasible Size", "Species Count", "Minimum Species Size", "Maximum Species Size",
             "Mean Species Size"]

AVG_keys = ["Instability", "X-Symmetry", "Y-Symmetry", "Z-Symmetry", "Surface Area",
            "Floor", "Walls", "Roof", "Building vs BB Volume Ratio", "BB vs Total Volume Ratio"]


def confidence_interval(values, confidence):
    return np.std(values) / np.sqrt(len(values)) * confidence
