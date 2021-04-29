import itertools
import random
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.decomposition import PCA
from tensorflow.python.keras.utils.np_utils import to_categorical
from Autoencoder import load_model, convert_to_integer, calculate_error, add_noise
from Visualization import voxel_plot, auto_encoder_plot
from Delenox_Config import value_range

# plt.style.use('seaborn')
flatten = itertools.chain.from_iterable

locations = [(0, 0), (0, 1), (1, 0), (1, 1)]
pca_locs = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
ae_label = ['Vanilla AE', 'De-Noising AE']
process_count = 8


def load_training_set(label):
    return [list(np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Training_Set.npz".format(label, i), allow_pickle=True)['arr_0'])[-1000:] for i in range(10)]


def load_populations(label):
    return [[list(np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Population_{}.npz".format(label, j, i), allow_pickle=True)['arr_0'].item().values()) for i in range(10)] for j in range(10)]


def load_metric(labels, metric):
    return [np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Metrics.npy".format(directory, 9), allow_pickle=True)[metric] for directory in labels]


def vector_entropy(vector1, population):
    return np.mean([entropy(vector1, neighbour) for neighbour in population])


def lattice_diversity(experiment, args=None):
    """
    Calculate the average population diversity for the given experiment using KL Divergence.
    :param experiment: experiment label to load populations
    :param args: unused - needed for modularity
    :return: mean diversity for each phase and confidence intervals
    """
    pool = Pool(process_count)
    experiment_population = load_populations(experiment)
    experiment_diversity = []
    for phase in range(len(experiment_population)):
        phase_diversities = []
        for population in range(len(experiment_population[phase])):
            print("Starting Experiment {} - Phase {} - Population {}".format(experiment, phase, population))
            lattices = [softmax(to_categorical(lattice).ravel()) for lattice in experiment_population[phase][population]]
            results = [pool.apply_async(vector_entropy, (lattice, lattices)) for lattice in lattices]
            phase_diversities.append(np.mean([result.get() for result in results]))
        experiment_diversity.append(phase_diversities)
    experiment_diversity = np.stack(experiment_diversity, axis=1)
    means = np.mean(experiment_diversity, axis=1)
    ci = np.std(experiment_diversity, axis=1) / np.sqrt(10) * 1.96
    pool.close()
    pool.join()
    return means, ci


def test_population(experiments):
    """

    :param experiments:
    :return:
    """
    test_pop = []
    for experiment in experiments:
        phases = load_populations(experiment)
        for populations in phases:
            for population in populations:
                for building in random.sample(population, 1):
                    test_pop.append(building)
    return test_pop


def reconstruction_accuracy(experiment, args):
    """

    :param experiment:
    :param args:
    :return:
    """
    means = []
    cis = []
    for phase in range(10):
        print("Loading Autoencoder from Phase {}".format(phase))
        try:
            encoder = load_model("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/encoder".format(experiment, phase))
            decoder = load_model("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/decoder".format(experiment, phase))
        except FileNotFoundError:
            encoder = load_model("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/encoder".format(experiment, 0))
            decoder = load_model("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/decoder".format(experiment, 0))
        errors = []
        for lattice in args[0]:
            if experiment[-3:] == 'DAE':
                compressed = encoder.predict(add_noise(lattice)[None])[0]
            else:
                compressed = encoder.predict(lattice[None])[0]
            reconstructed = decoder.predict(compressed[None])[0]

            auto_encoder_plot(convert_to_integer(lattice), compressed, convert_to_integer(reconstructed), calculate_error(lattice, reconstructed))
            errors.append(calculate_error(lattice, reconstructed))
        means.append(np.mean(errors))
        cis.append(np.std(errors) / np.sqrt(len(errors)) * 1.96)
    return np.asarray(means), np.asarray(cis)


def pca_population(experiments):
    """

    :param experiments:
    :return:
    """
    pca = PCA(n_components=2)
    print("Loading experiment training sets and flattening them into 1D arrays...")
    pca_pop = [[[convert_to_integer(lattice).ravel() for lattice in phase] for phase in load_training_set(experiment)] for experiment in experiments]
    pca.fit(list(flatten(list(flatten(pca_pop)))))
    return pca, pca_pop


def pca_graph(experiment, args=None):
    """

    :param experiment:
    :param args:
    :return:
    """
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 7), sharex=True, sharey=True)
    print("PCA Scatter Plots - {}".format(experiment))
    fig.suptitle("PCA Scatter Plots - {}".format(experiment))
    experiment_population = args[0][1][args[1]]
    diversity = []
    for training_set in range(len(experiment_population)):
        principal_components = args[0][0].transform(experiment_population[training_set])
        diversity.append(np.asarray([np.mean([np.linalg.norm(pc - other) for other in principal_components]) for pc in principal_components]))
        axis = axes[pca_locs[training_set][0]][pca_locs[training_set][1]]
        axis.set_title("Phase {}".format(training_set + 1))
        axis.scatter([item[0] for item in principal_components], [item[1] for item in principal_components], s=10, alpha=0.5, label="Phase {:d}".format(training_set + 1))
    plt.setp(axes[-1, :], xlabel='PC1')
    plt.setp(axes[:, 0], ylabel='PC2')
    fig.tight_layout()
    fig.show()
    return np.asarray([np.mean(diversity1) for diversity1 in diversity]), np.asarray([np.std(diversity1) / np.sqrt(len(diversity1)) * 1.96 for diversity1 in diversity])


# TODO: Modular version of the NEAT metrics visualizations for grid_plot
def neat_metric(experiment, metric):
    metric = np.asarray(metric_list[counter].item()[key])

    if metric.shape == (10, 1000):
        metric = np.stack(metric, axis=1)

    generations = range(len(metric))

    mean = np.mean(metric[generations], axis=1)
    ci = np.std(metric[generations], axis=1) / np.sqrt(10) * 1.96
    pass


def grid_plot(experiments, function, title, args=None):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9), sharex=True, sharey=True)
    fig.suptitle("{} over {:d} Runs.".format(title, 10))
    baseline_means, baseline_ci = function(experiments[0], (args, 0))
    counter = 0
    for experiment in [(1, ), (2, 3), (4, 5), (6, 7)]:
        axis = axes[locations[counter][0]][locations[counter][1]]
        axis.errorbar(x=range(10), y=baseline_means, label=labels[0], color=colors[0])
        axis.fill_between(x=range(10), y1=baseline_means + baseline_ci, y2=baseline_means - baseline_ci, color=colors[0], alpha=0.1)
        axis.set_title(labels[experiment[0]][:-3] + " Autoencoders")
        for sub in range(len(experiment)):
            means, ci = function(experiments[experiment[sub]], (args, experiment[sub]))
            axis.errorbar(x=range(10), y=means, label=ae_label[sub], color=colors[sub + 1])
            axis.fill_between(x=range(10), y1=means + ci, y2=means - ci, color=colors[sub + 1], alpha=0.1)
        axis.legend()
        axis.grid()
        counter += 1
    plt.setp(axes[-1, :], xlabel='Phase')
    plt.setp(axes[:, 0], ylabel=title)
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

    labels = ["Static AE", "Random AE",
              "Full History AE", "Full History DAE",
              "Latest Set AE", "Latest Set DAE",
              "Novelty Archive AE", "Novelty Archive DAE"]
    colors = ['black', 'red', 'blue', 'green', 'brown', 'orange', 'purple', 'cyan']
    keys = ["Node Complexity", "Connection Complexity", "Archive Size", "Best Novelty", "Mean Novelty"]

    # pca, pca_pop = pca_population(labels)
    # grid_plot(labels, pca_graph, "Diversity of Populations' Principle Components", args=(pca, pca_pop))

    test_pop = test_population(labels)
    grid_plot(labels, reconstruction_accuracy, "Reconstruction Error", args=test_pop)
