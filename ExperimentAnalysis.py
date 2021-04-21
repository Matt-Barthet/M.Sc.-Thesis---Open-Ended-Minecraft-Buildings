import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tensorflow.python.keras.utils.np_utils import to_categorical
from mpl_toolkits.mplot3d import axes3d, Axes3D  #<-- Note the capitalization!
import time
from Autoencoder import load_model, convert_to_integer, test_accuracy, create_auto_encoder, auto_encoder_3d
from Constraints import apply_constraints
from Delenox_Config import value_range
from Visualization import voxel_plot, plot_statistics
import itertools
# plt.style.use('seaborn')
flatten = itertools.chain.from_iterable

locations = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def pca_buildings(phases):

    populations = [[np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Training_Set.npy".format(label, i), allow_pickle=True)[:-250] for i in range(10)] for label in labels]
    converted_population = [[[convert_to_integer(lattice).ravel() for lattice in phase] for phase in population] for population in populations]
    pca = PCA(n_components=2)
    pca.fit(list(flatten(list(flatten(converted_population)))))
    eucl_df = pd.DataFrame({'Phase': [], 'Average Euclidean Distance': [], 'Experiment': []})

    for population in range(len(converted_population)):
        print("PCA on Population {:d}".format(population))
        plt.figure()
        plt.title("Novel Set PCA - {}".format(labels[population]))
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        eucl_averages = []
        eucl_std = []

        pca_df = pd.DataFrame({'x': [], 'y': [], 'Phase': []})

        for phase in phases:

            principal_components = pca.transform(converted_population[population][phase])
            principal_df = pd.DataFrame({'x': [item[0] for item in principal_components], 'y': [item[1] for item in principal_components], 'Phase': np.repeat("{:d}".format(phase + 1), len(principal_components))})
            pca_df = pd.concat([pca_df, principal_df], axis=0)

            averages = []
            for vector in principal_components:
                average = 0
                for other in principal_components:
                    dist = np.linalg.norm(vector - other)
                    average = np.mean([average, dist])
                averages.append(average)
            eucl_averages.append(np.round(np.mean(averages), 2))
            eucl_std.append(np.std(averages))

            for average in averages:
                tmp = pd.DataFrame({'Phase': [phase + 1],
                                    'Average Euclidean Distance': [average],
                                    'Experiment': [labels[population]]})
                eucl_df = pd.concat([eucl_df, tmp], axis=0)

            plt.scatter(principal_df['x'],
                        principal_df['y'],
                        cmap=[phase] * len(principal_df),
                        s=10,
                        alpha=0.5,
                        label="Phase {:d}".format(phase + 1))

        plt.xlim(-60, 105)
        plt.ylim(-70, 80)
        plt.legend()

        # Use seaborn for easy faceting
        g = sns.FacetGrid(pca_df, col="Phase", hue="Phase")
        g.fig.suptitle("Novel Training Set PCA - {}".format(labels[population]))
        g = (g.map(plt.scatter, "x", "y", edgecolor="w", s=10, alpha=0.5))
        plt.subplots_adjust(left=0.032, right=0.992, top=0.836, bottom=0.143)

    """plt.figure()
    plt.subplots_adjust(left=0.080, right=0.790, top=0.915, bottom=0.090)
    ax = sns.lineplot(x='Phase', y='Average Euclidean Distance', hue='Experiment', data=eucl_df, ci='sd')
    ax.set_title('Average PW Euclidean Distance using PCA')
    ax.legend(frameon=True, bbox_to_anchor=(1.005, 0.65), loc="upper left")
    legend = ax.get_legend()
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.show()"""

    plt.figure()
    plt.title("PCA Diversity - Regression Lines")
    plt.xlabel("Phase")
    plt.ylabel("Average Euclidean Distance")

    for label in labels:
        subset = eucl_df[eucl_df['Experiment'] == label]
        linear_regressor = LinearRegression()
        linear_regressor.fit(subset['Phase'].values.reshape(-1, 1), subset['Average Euclidean Distance'].values.reshape(-1, 1))
        Y_pred = linear_regressor.predict(subset['Phase'].values.reshape(-1, 1))
        plt.plot(subset['Phase'], Y_pred)

    plt.show()


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
    return [np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Training_Set.npy".format(label, i), allow_pickle=True)[:100] for i in range(10)]


def load_populations(label):
    return [[list(np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Population_{}.npy".format(label, j, i), allow_pickle=True).item().values())for i in range(10)] for j in range(10)]


def lattice_diversity(experiment):

    pool = Pool(16)
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
    return means, ci


def population_diversity(experiments):

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle("Population Diversity over {:d} Runs.".format(10))

    baseline_means, baseline_ci = lattice_diversity(experiments[0])

    for experiment in range(1, len(experiments)):
        means, ci = lattice_diversity(experiments[experiment])
        axis = axes[locations[experiment - 1][0]][locations[experiment - 1][1]]
        axis.errorbar(x=range(10), y=baseline_means, label=labels[0], color=colors[0])
        axis.errorbar(x=range(10), y=means, label=labels[experiment], color=colors[experiment])
        axis.fill_between(x=range(10), y1=baseline_means + baseline_ci, y2=baseline_means - baseline_ci, color=colors[0], alpha=0.1)
        axis.fill_between(x=range(10), y1=means + ci, y2=means - ci, color=colors[experiment], alpha=0.1)

        axis.legend()
        axis.grid()

    plt.setp(axes[-1, :], xlabel='Phase')
    plt.setp(axes[:, 0], ylabel='KL Divergence')
    plt.tight_layout()
    plt.show()


def plot_metric(experiments, function, title):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle("{} over {:d} Runs.".format(title, 10))

    baseline_means, baseline_ci = function(experiments[0])

    for experiment in range(1, len(experiments)):
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
    plt.tight_layout()
    plt.show()


def novel_diversity(populations):

    eucl_df = pd.DataFrame({'Phase': [], 'Eucl. Distance': [], 'Experiment': []})
    entropy_df = pd.DataFrame({'Phase': [], 'Entropy': [], 'Experiment': []})

    _, encoder, decoder = create_auto_encoder(auto_encoder_3d,
                                              noisy=None,
                                              phase=0,
                                              save=False,
                                              experiment=None)
    pool = Pool(4)

    for population in range(len(populations)):
        print("Starting Population {:d}".format(population))
        for phase in range(len(populations[population])):
            print("Starting Phase {:d}".format(phase))

            results = []
            phase_vectors = []

            for lattice in populations[population][phase]:
                compressed = encoder.predict(lattice[None])[0]
                compressed = softmax(compressed)
                phase_vectors.append(compressed)

            for vectors in phase_vectors:
                results.append(pool.apply_async(vector_novelty, (vectors, phase_vectors)))

            for result in results:
                tmp = pd.DataFrame({'Phase': [phase + 1],
                                    'Eucl. Distance': [result.get()],
                                    'Experiment': [labels[population]]})
                eucl_df = pd.concat([eucl_df, tmp], axis=0)

            results.clear()

            for vectors in phase_vectors:
                results.append(pool.apply_async(vector_entropy, (vectors, phase_vectors)))

            for result in results:
                tmp = pd.DataFrame({'Phase': [phase + 1],
                                    'Entropy': [result.get()],
                                    'Experiment': [labels[population]]})
                entropy_df = pd.concat([entropy_df, tmp], axis=0)

    plt.figure()
    plt.subplots_adjust(left=0.080, right=0.790, top=0.915, bottom=0.090)
    ax = sns.lineplot(x='Phase', y='Eucl. Distance', hue='Experiment', data=eucl_df, ci='sd')
    ax.set_title('Average Pairwise Eucl. Distance of Latent Vectors Training Sets')
    ax.legend(frameon=True, bbox_to_anchor=(1.005, 0.65), loc="upper left")
    legend = ax.get_legend()
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.show()

    plt.figure()
    plt.subplots_adjust(left=0.080, right=0.790, top=0.915, bottom=0.090)
    ax = sns.lineplot(x='Phase', y='Entropy', hue='Experiment', data=entropy_df, ci='sd')
    ax.set_title('Average Entropy of Latent Vectors in Training Sets')
    ax.legend(frameon=True, bbox_to_anchor=(1.005, 0.65), loc="upper left")
    legend = ax.get_legend()
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.show()


def vector_novelty(vector1, population):
    diversities = []
    for neighbour in population:
        if not np.array_equal(vector1, neighbour):
            diversity = 0
            for element in range(len(neighbour)):
                diversity += np.square(vector1[element] - neighbour[element])
            diversities.append(np.sqrt(diversity))
    return np.mean(diversities)


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
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    subset_size = 1000
    labels = ["Static AE", "Random AE", "Full History AE", "Full History DAE", "Latest Set AE", "Latest Set DAE", "Novelty Archive AE"]
    colors = ['black', 'red', 'blue', 'green', 'brown', 'orange', 'purple']
    keys = ["Node Complexity", "Connection Complexity", "Archive Size", "Best Novelty", "Mean Novelty"]

    population_diversity(labels)

    # novel_diversity(training_set)
    # pca_buildings(range(10))
    # accuracy_plot(training_set, labels)
    # plot_metric(labels, colors, keys)
    # lattice_diversity(labels)


