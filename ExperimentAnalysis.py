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

from Autoencoder import load_model, convert_to_integer, calculate_error

# plt.style.use('seaborn')
flatten = itertools.chain.from_iterable

locations = [(0, 0), (0, 1), (1, 0), (1, 1)]
pca_locs = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
ae_label = ['Vanilla AE', 'De-Noising AE']
test_pop = []
process_count = 8


def load_training_set(label):
    return [list(np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Training_Set.npz".format(label, i), allow_pickle=True)['arr_0'])[-1000:] for i in range(10)]


def load_populations(label):
    return [[list(np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Population_{}.npz".format(label, j, i), allow_pickle=True)['arr_0'].item().values()) for i in range(10)] for j in range(10)]


def vector_entropy(vector1, population):
    return np.mean([entropy(vector1, neighbour) for neighbour in population])


def lattice_diversity(experiment):
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
    for experiment in experiments:
        phases = load_populations(experiment)
        for populations in phases:
            for population in populations:
                for building in random.sample(population, 10):
                    test_pop.append(building)


# TODO: Test this function.
def reconstruction_accuracy(experiment):
    means = []
    cis = []
    for phase in range(10):
        try:
            encoder = load_model("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/encoder".format(experiment, phase))
            decoder = load_model("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/decoder".format(experiment, phase))
        except FileNotFoundError:
            encoder = load_model("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/encoder".format(experiment, 0))
            decoder = load_model("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/decoder".format(experiment, 0))
        errors = []
        for lattice in test_pop:
            compressed = encoder.predict(lattice[None])[0]
            reconstructed = decoder.predict(compressed[None])[0]
            integer_reconstruct = convert_to_integer(reconstructed)
            errors.append(calculate_error(lattice, integer_reconstruct))
        means.append(np.mean(errors))
        cis.append(np.std(errors) / np.sqrt(len(errors)) * 1.96)
    return means, cis


# TODO: Modular version of the PCA function for grid_plot
def pca_graphs(experiments):
    converted_population = []
    for experiment in experiments:
        training_set = load_training_set(experiment)
        converted_population.append([[convert_to_integer(lattice).ravel() for lattice in phase] for phase in training_set])

    pca = PCA(n_components=2)
    pca.fit(list(flatten(list(flatten(converted_population)))))

    diversity_fig, diversity_axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharex=True, sharey=True)
    diversity_fig.suptitle("Diversity of PCA Values in Experiment Training Sets")
    baseline_mean = 0
    baseline_ci = 0

    for experiment in range(len(experiments)):
        print("PCA Scatter Plots for {}".format(experiments[experiment]))
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 8), sharex=True, sharey=True)
        fig.suptitle("PCA Scatter Plots - {}".format(experiments[experiment]))
        experiment_population = converted_population[experiment]

        means = []
        ci = []
        for training_set in range(len(experiment_population)):
            principal_components = pca.transform(experiment_population[training_set])

            phase_mean = []
            for pc in principal_components:
                mean = []
                for other in principal_components:
                    mean.append(np.linalg.norm(pc - other))
                phase_mean.append(np.mean(mean))

            means.append(np.mean(phase_mean))
            ci.append(np.std(phase_mean) / np.sqrt(len(phase_mean)) * 1.96)

            axis = axes[pca_locs[training_set][0]][pca_locs[training_set][1]]
            axis.set_title("Phase {}".format(training_set + 1))
            axis.scatter([item[0] for item in principal_components],
                        [item[1] for item in principal_components],
                        s=10,
                        alpha=0.5,
                        label="Phase {:d}".format(training_set + 1))

        plt.setp(axes[-1, :], xlabel='PC1')
        plt.setp(axes[:, 0], ylabel='PC2')
        fig.tight_layout()
        fig.show()

        means = np.asarray(means)
        ci = np.asarray(ci)

        if experiment == 0:
            baseline_mean = means
            baseline_ci = ci
        else:
            axis = diversity_axes[locations[experiment - 1][0]][locations[experiment - 1][1]]
            axis.errorbar(x=range(10), y=baseline_mean, label=labels[0], color=colors[0])
            axis.errorbar(x=range(10), y=means, label=labels[experiment], color=colors[experiment])
            axis.fill_between(x=range(10), y1=baseline_mean + baseline_ci, y2=baseline_mean - baseline_ci,
                              color=colors[0], alpha=0.1)
            axis.fill_between(x=range(10), y1=means + ci, y2=means - ci, color=colors[experiment], alpha=0.1)
            axis.legend()
            axis.grid()

    plt.setp(diversity_axes[-1, :], xlabel='Phase')
    plt.setp(diversity_axes[:, 0], ylabel="Mean Eucl. Distance")
    diversity_fig.tight_layout()
    diversity_fig.show()


# TODO: Modular version of the NEAT metrics visualizations for grid_plot
def neat_metric(experiments, metric):
    pass


def grid_plot(experiments, function, title):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9), sharex=True, sharey=True)
    fig.suptitle("{} over {:d} Runs.".format(title, 10))
    baseline_means, baseline_ci = function(experiments[0])
    counter = 0
    for experiment in [(1, ), (2, 3), (4, 5), (6, 7)]:
        axis = axes[locations[counter][0]][locations[counter][1]]
        axis.errorbar(x=range(10), y=baseline_means, label=labels[0], color=colors[0])
        axis.fill_between(x=range(10), y1=baseline_means + baseline_ci, y2=baseline_means - baseline_ci, color=colors[0], alpha=0.1)
        axis.set_title(labels[experiment[0]][:-3] + " Autoencoders")
        for sub in range(len(experiment)):
            means, ci = function(experiments[experiment[sub]])
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

    labels = ["Static AE", "Random AE", "Full History AE", "Full History DAE", "Latest Set AE", "Latest Set DAE", "Novelty Archive AE", "Novelty Archive DAE"]
    colors = ['black', 'red', 'blue', 'green', 'brown', 'orange', 'purple', 'cyan']
    keys = ["Node Complexity", "Connection Complexity", "Archive Size", "Best Novelty", "Mean Novelty"]

    # pca_graphs(labels)
    grid_plot(labels, lattice_diversity, "Population Diversity")
    test_population(labels)

