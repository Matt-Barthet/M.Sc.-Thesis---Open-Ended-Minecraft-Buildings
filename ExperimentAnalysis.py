import bz2
import itertools
import os
import pickle as pkl
from multiprocessing import Pool
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.decomposition import PCA
from tensorflow.python.keras.utils.np_utils import to_categorical

from Autoencoder import load_model, convert_to_integer, test_accuracy

# plt.style.use('seaborn')
flatten = itertools.chain.from_iterable

locations = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def load_and_compress(labels):
    for label in labels:
        for phase in range(10):

            try:
                training_set = np.load("D:/Persistent Archive Tests/{}/Phase{}/Training_Set.npy".format(label, phase), allow_pickle=True)
                np.savez_compressed("D:/Persistent Archive Tests/{}/Phase{}/Training_Set.npz".format(label, phase), training_set)
                os.remove("D/Persistent Archive Tests/{}/Phase{}/Training_Set.npy".format(label, phase))
            except FileNotFoundError:
                pass

            try:
                for population in range(10):
                    pop = pkl.load(open("D:/Persistent Archive Tests/{}/Phase{}/Neat_Population_{}.pkl".format(label, phase, population), 'rb'))
                    sfile = bz2.BZ2File("D:/Persistent Archive Tests/{}/Phase{}/Neat_Population_{}.bz2".format(label, phase, population), 'wb')
                    pkl.dump(pop, sfile)
                    os.remove("D:/Persistent Archive Tests/{}/Phase{}/Neat_Population_{}.pkl".format(label, phase, population))
            except FileNotFoundError:
                pass

            for population in range(10):
                pop = np.load("D:/Persistent Archive Tests/{}/Phase{}/Population_{}.npy".format(label, phase, population), allow_pickle=True)
                np.savez_compressed("D:/Persistent Archive Tests/{}/Phase{}/Population_{}.npz".format(label, phase, population), pop)
                os.remove("D:/Persistent Archive Tests/{}/Phase{}/Population_{}.npy".format(label, phase, population))


def accuracy_plot(populations, models):
    df = pd.DataFrame({'Phase': [], 'Reconstruction Error': [], 'Experiment': []})

    for population in range(len(models)):

        for phase in range(len(populations)):
            print("Starting new phase")

            if population != 0:
                encoder = load_model("{}Phase{}/encoder".format(models[population], 9))
                decoder = load_model("{}Phase{}/decoder".format(models[population], 9))
            else:
                encoder = load_model("{}Phase{}/encoder".format(models[population], 0))
                decoder = load_model("{}Phase{}/decoder".format(models[population], 0))

            errors = test_accuracy(encoder, decoder, populations[population][phase], mean=False)

            for error in errors:
                tmp = pd.DataFrame({'Phase': [phase + 1],
                                    'Reconstruction Error': [error],
                                    'Experiment': [labels[population]]})
                df = pd.concat([df, tmp], axis=0)

    plt.figure()
    plt.subplots_adjust(left=0.080, right=0.790, top=0.915, bottom=0.090)
    ax = sns.lineplot(x='Phase', y='Reconstruction Error', hue='Experiment', data=df, ci='sd')
    ax.set_title('Average Reconstruction Error using Final AE')
    ax.legend(frameon=True, bbox_to_anchor=(1.005, 0.65), loc="upper left")
    legend = ax.get_legend()
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.show()


def load_training_set(label):
    return [list(np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Training_Set.npz".format(label, i), allow_pickle=True)['arr_0'])[-1000:] for i in range(10)]

def load_populations(label):
    return [[list(np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Population_{}.npz".format(label, j, i), allow_pickle=True)['arr_0'].item().values()) for i in range(10)] for j in range(10)]


def lattice_diversity(experiment):
    pool = Pool(12)
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
    test_pop = []
    for experiment in experiments:
        phases = load_populations(experiment)
        for populations in phases:
            for population in populations:
                for building in random.sample(population, 10):
                    test_pop.append(building)
    return test_pop

def reconstruction_accuracy(experiment):

    pass


pca_locs = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]

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


def grid_plot(experiments, function, title):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle("{} over {:d} Runs.".format(title, 10))
    baseline_means, baseline_ci = function(experiments[0])
    for experiment in range(1, len( )):
        means, ci = function(experiments[experiment])
        axis = axes[locations[experiment - 1][0]][locations[experiment - 1][1]]
        axis.errorbar(x=range(10), y=baseline_means, label=labels[0], color=colors[0])
        axis.errorbar(x=range(10), y=means, label=labels[experiment], color=colors[experiment])
        axis.fill_between(x=range(10), y1=baseline_means + baseline_ci, y2=baseline_means - baseline_ci, color=colors[0], alpha=0.1)
        axis.fill_between(x=range(10), y1=means + ci, y2=means - ci, color=colors[experiment], alpha=0.1)
        axis.legend()
        axis.grid()
    plt.setp(axes[-1, :], xlabel='Phase')
    plt.setp(axes[:, 0], ylabel=title)
    fig.tight_layout()
    fig.show()


def vector_entropy(vector1, population):
    return np.mean([entropy(vector1, neighbour) for neighbour in population])


def plot_metric(labels, colors, keys):

    metric_list = [np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Metrics.npy".format(directory, 9), allow_pickle=True) for directory in labels]

    for key in keys:

        plt.figure()
        plt.title("{} vs Generation over {:d} Runs.".format(key, 10))
        plt.xlabel("Generation")
        plt.ylabel(key)

        for counter in range(len(metric_list)):

            metric = np.asarray(metric_list[counter].item()[key])

            if metric.shape == (10, 1000):
                metric = np.stack(metric, axis=1)

            generations = range(len(metric))

            mean = np.mean(metric[generations], axis=1)
            ci = np.std(metric[generations], axis=1) / np.sqrt(10) * 1.96

            # Plotting the mean of given metric over generations
            plt.errorbar(x=generations,
                         y=mean,
                         fmt='-',
                         label=labels[counter],
                         alpha=1,
                         color=colors[counter])

            # Filling the deviation from the mean in a translucent color.
            plt.fill_between(x=generations,
                             y1=mean + ci,
                             y2=mean - ci,
                             color=colors[counter],
                             alpha=0.1)

        legend = plt.legend(frameon=True)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
    plt.show()


if __name__ == '__main__':

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    subset_size = 1000
    labels = ["Static AE", "Random AE", "Full History AE", "Full History DAE", "Latest Set AE", "Latest Set DAE", "Novelty Archive AE"]
    colors = ['black', 'red', 'blue', 'green', 'brown', 'orange', 'purple']
    keys = ["Node Complexity", "Connection Complexity", "Archive Size", "Best Novelty", "Mean Novelty"]

    # load_and_compress(labels)
    # grid_plot(labels, pca_graphs, "PCA")
    # pca_graphs(labels)
    test_population(labels)

    # accuracy_plot(training_set, labels)
    # plot_metric(labels, colors, keys)
    # lattice_diversity(labels)


