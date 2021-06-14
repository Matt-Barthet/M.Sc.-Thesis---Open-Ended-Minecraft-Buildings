import itertools
import random
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.decomposition import PCA
from tensorflow.python.keras.utils.np_utils import to_categorical
from Autoencoder import load_model, convert_to_integer, calculate_error, add_noise
from Visualization import voxel_plot, auto_encoder_plot, get_color_map, expressive_graph
from Delenox_Config import value_range
from Constraints import *
from NeatGenerator import novelty_search
import os

flatten = itertools.chain.from_iterable

locations = [(0, 0), (0, 1), (1, 0), (1, 1)]
pca_locs = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
novelty_spectrum_subplots = [[(5, 5, index * 5 + offset) for index in range(5)] for offset in range(1, 6)]
ae_label = ['Vanilla AE', 'Denoising AE']
process_count = 16


def load_training_set(label):
    return [list(
        np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Training_Set.npz".format(label, i),
                allow_pickle=True)['arr_0'])[-1000:] for i in range(10)]


def load_populations(label):
    return [[list(
        np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Population_{}.npz".format(label, j, i),
                allow_pickle=True)['arr_0'].item().values()) for i in range(10)] for j in range(10)]


def load_metric(label, metric):
    try:
        return np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Metrics.npy".format(label, 9),
                       allow_pickle=True).item()[metric]
    except FileNotFoundError:
        return np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Metrics.npz".format(label, 9),
                       allow_pickle=True)['arr_0'].item()[metric]


def load_metrics(labels, metric):
    return [np.load("Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/Metrics.npy".format(directory, 9),
                    allow_pickle=True)[metric] for directory in labels]


def load_autoencoder(label, phase):
    try:
        encoder = load_model(
            "Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/encoder".format(label, phase))
        decoder = load_model(
            "Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/decoder".format(label, phase))
    except FileNotFoundError:
        encoder = load_model(
            "Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/encoder".format(label, 0))
        decoder = load_model(
            "Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/decoder".format(label, 0))
    return encoder, decoder


def vector_entropy(vector1, population):
    return np.mean([entropy(vector1, neighbour) for neighbour in population])


def lattice_diversity(experiment, args=None):
    """
    Calculate the average population diversity for the given experiment using KL Divergence.
    :param experiment: experiment label to load populations
    :param args: unused - needed for modularity
    :return: mean diversity for each phase and confidence intervals
    """
    experiment_population = load_populations(experiment)
    experiment_diversity = []
    for phase in range(len(experiment_population)):
        phase_diversities = []
        for population in range(len(experiment_population[phase])):
            print("Starting Experiment {} - Phase {} - Population {}".format(experiment, phase, population))
            lattices = [softmax(to_categorical(lattice).ravel()) for lattice in
                        experiment_population[phase][population]]
            results = [pool.apply_async(vector_entropy, (lattice, lattices)) for lattice in lattices]
            phase_diversities.append(np.mean([result.get() for result in results]))
        experiment_diversity.append(phase_diversities)
    experiment_diversity = np.stack(experiment_diversity, axis=1)
    means = np.mean(experiment_diversity, axis=1)
    ci = np.std(experiment_diversity, axis=1) / np.sqrt(10) * 1.96
    return means, ci


def diversity_from_humans(experiment, args=None):
    """
    Calculate the average population diversity for the given experiment using KL Divergence.
    :param experiment: experiment label to load populations
    :param args: unused - needed for modularity
    :return: mean diversity for each phase and confidence intervals
    """
    experiment_population = load_populations(experiment)
    human_population = np.load("Real-World Datasets/Ahousev5_Buildings_Varied.npy", allow_pickle=True)
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


def test_population(experiments):
    """

    :param experiments:
    :return:
    """
    test_pop = []
    for experiment in experiments:
        populations = load_training_set(experiment)
        for population in populations:
            for building in random.sample(population, 30):
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
            encoder = load_model(
                "Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/encoder".format(experiment, phase))
            decoder = load_model(
                "Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/decoder".format(experiment, phase))
        except FileNotFoundError:
            encoder = load_model(
                "Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/encoder".format(experiment, 0))
            decoder = load_model(
                "Delenox_Experiment_Data/Persistent Archive Tests/{}/Phase{}/decoder".format(experiment, 0))
        errors = []
        for lattice in args[0]:
            if experiment[-3:] == 'DAE':
                compressed = encoder.predict(add_noise(lattice)[None])[0]
            else:
                compressed = encoder.predict(lattice[None])[0]
            reconstructed = decoder.predict(compressed[None])[0]
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
    pca_pop = [[[convert_to_integer(lattice).ravel() for lattice in phase] for phase in load_training_set(experiment)]
               for experiment in experiments]
    pca.fit(list(flatten(list(flatten(pca_pop)))))
    return pca, pca_pop


def pca_graph(experiment, args=None, shareAxes=True):
    """

    :param shareAxes:
    :param experiment:
    :param args:
    :return:
    """
    print("PCA Scatter Plots - {}".format(experiment))
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(9, 5.25), sharex=shareAxes, sharey=shareAxes)
    fig.suptitle("PCA Scatter Plots - {}".format(experiment))
    experiment_population = args[0][1][args[1]]
    diversity = []
    for training_set in range(len(experiment_population)):
        principal_components = args[0][0].transform(experiment_population[training_set])
        diversity.append(np.asarray(
            [np.mean([np.linalg.norm(pc - other) for other in principal_components]) for pc in principal_components]))
        axis = axes[pca_locs[training_set][0]][pca_locs[training_set][1]]
        axis.set_title("Phase {}".format(training_set + 1),fontsize=12)
        axis.scatter([item[0] for item in principal_components], [item[1] for item in principal_components], s=7,
                     alpha=0.5, label="Phase {:d}".format(training_set + 1))
    plt.setp(axes[-1, :], xlabel='PC1')
    plt.setp(axes[:, 0], ylabel='PC2')
    fig.tight_layout()
    fig.savefig("../PCA {}.png".format(experiment))
    fig.show()
    return np.asarray([np.mean(diversity1) for diversity1 in diversity]), np.asarray(
        [np.std(diversity1) / np.sqrt(len(diversity1)) * 1.96 for diversity1 in diversity])


# TODO: Modular version of the NEAT metrics visualizations for grid_plot
def neat_metric(experiment, metric):
    metric = np.asarray(load_metric(experiment, metric[0]))
    if metric.shape == (10, 1000):
        metric = np.stack(metric, axis=1)
    generations = range(len(metric))
    mean = np.mean(metric[generations], axis=1)
    ci = np.std(metric[generations], axis=1) / np.sqrt(10) * 1.96
    return mean, ci


plt.rc('axes', labelsize=12)


def grid_plot(experiments, function, title, args=None, shareAxes=True):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6.5), sharex=shareAxes, sharey=shareAxes)
    baseline_means, baseline_ci = function(experiments[0], (args, 0))
    counter = 0
    for experiment in [(1,), (2, 3), (4, 5), (6, 7)]:
        axis = axes[locations[counter][0]][locations[counter][1]]
        axis.errorbar(x=range(len(baseline_means)), y=baseline_means, label=labels[0], color=colors[0])
        axis.fill_between(x=range(len(baseline_means)), y1=baseline_means + baseline_ci,
                          y2=baseline_means - baseline_ci, color=colors[0], alpha=0.1)
        axis.set_title(labels[experiment[0]][:-3] + " Autoencoders", fontsize=12)
        for sub in range(len(experiment)):
            means, ci = function(experiments[experiment[sub]], (args, experiment[sub]))
            axis.errorbar(x=range(len(means)), y=means, label=ae_label[sub], color=colors[sub + 1])
            axis.fill_between(x=range(len(means)), y1=means + ci, y2=means - ci, color=colors[sub + 1], alpha=0.1)
        axis.grid()
        counter += 1

        handles, legendlabels = axis.get_legend_handles_labels()

    fig.legend(handles=handles, labels=legendlabels, fontsize=12, loc='lower center', ncol=6, )
    plt.setp(axes[-1, :], xlabel='Phase')
    plt.setp(axes[:, 0], ylabel=title)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.savefig("../{}.png".format(title))
    fig.show()


def draw_lines_fig(fig):
    line = plt.Line2D((0.12, 0.12), (.1, .9), color="k", linewidth=2)
    fig.add_artist(line)
    line = plt.Line2D((0.91, 0.91), (.1, .9), color="k", linewidth=2)
    fig.add_artist(line)
    line = plt.Line2D((.275, .275), (.1, .9), color="k", linewidth=2)
    fig.add_artist(line)
    line = plt.Line2D((.4325, .4325), (.1, .9), color="k", linewidth=2)
    fig.add_artist(line)
    line = plt.Line2D((.595, .595), (.1, .9), color="k", linewidth=2)
    fig.add_artist(line)
    line = plt.Line2D((.7575, .7575), (.1, .9), color="k", linewidth=2)
    fig.add_artist(line)
    return fig


def novelty_spectrum(labels):
    xlabels = ['Most\nNovel', 'Upper\nQuartile', 'Median\nNovel', 'Lower\nQuartile', 'Least\nNovel']
    for experiment in labels:
        print("Starting Experiment {}".format(experiment))
        phases = load_training_set(experiment)
        fig = draw_lines_fig(plt.figure(figsize=(12, 12)))
        fig.suptitle("Range of Generated Content - {}".format(experiment), fontsize=18)

        for phase in range(1, len(phases), 2):
            print("Starting Phase {}".format(phase))
            encoder, _ = load_autoencoder(experiment, phases[phase])

            # phase_pop = random.sample(phases[phase], 250)
            phase_pop = phases[phase]

            original = {lattice_id: convert_to_integer(lattice) for (lattice_id, lattice) in enumerate(phase_pop)}

            if experiment[-3:] == 'DAE':
                compressed = {lattice_id: encoder.predict(add_noise(lattice)[None])[0] for (lattice_id, lattice) in
                              enumerate(phases[phase])}
            else:
                compressed = {lattice_id: encoder.predict(lattice[None])[0] for (lattice_id, lattice) in
                              enumerate(phases[phase])}

            fitness = {}
            jobs = []
            for key in compressed.keys():
                parameters = (key, compressed, {})
                jobs.append(pool.apply_async(novelty_search, parameters))
            for job, genome_id in zip(jobs, compressed.keys()):
                fitness.update({genome_id: job.get()})

            sorted_keys = [k for k, _ in sorted(fitness.items(), key=lambda item: item[1])]

            for number, plot in enumerate(np.linspace(len(sorted_keys) - 1, 0, 5, dtype=int)):
                ax = fig.add_subplot(novelty_spectrum_subplots[int(phase / 2)][number][0],
                                     novelty_spectrum_subplots[int(phase / 2)][number][1],
                                     novelty_spectrum_subplots[int(phase / 2)][number][2], projection='3d')

                ax.voxels(original[sorted_keys[plot]], edgecolor="k", facecolors=get_color_map(original[sorted_keys[plot]], 'blue'))
                ax.set_axis_off()
                if phase == 1:
                    ax.text(-37, 0, -5, s=xlabels[number], fontsize=15)
                if number == 4:
                    ax.text(5, 3, -40, s='Phase {}'.format(phase+1), fontsize=15)
        fig.show()


def expressive_analysis(experiments):
    """

    :param experiments:
    :param metric1:
    :param metric2:
    :return:
    """
    for experiment in experiments:
        phases = load_training_set(experiment)
        counter = 0
        for phase in [phases[0], phases[-1]]:
            surface_areas = []
            stabilities = []

            converted = [convert_to_integer(lattice) for lattice in phase]

            print("Starting Analysis")
            for lattice in converted:
                horizontal_bounds, depth_bounds, vertical_bounds = bounding_box(lattice)
                width = (horizontal_bounds[1] - horizontal_bounds[0])
                height = vertical_bounds[1]
                depth = (depth_bounds[1] - depth_bounds[0])
                lattice_stability, floor_stability = stability(lattice)
                stabilities.append(floor_stability)
                roof_count = 0
                walls = 0
                floor_count = 0
                interior_count = 0
                total_count = 0
                for (x, y, z) in value_range:
                    if lattice[x][y][z] == 0:
                        continue
                    total_count += 1
                    if lattice[x][y][z] == 1:
                        interior_count += 1
                    elif lattice[x][y][z] == 2:
                        walls += 1
                    elif lattice[x][y][z] == 4:
                        roof_count += 1
                    elif lattice[x][y][z] == 3:
                        floor_count += 1
                surface_areas.append((walls + roof_count + floor_count) / (2 * (width + depth + height)))

            print("Plotting Expressive Graph")
            expressive_graph(surface_areas, stabilities,
                             "Expressive Analysis - {}: Phase {}".format(experiment, counter),
                             "Surface Area / Bounding Box Area", "Lateral Stability")
            counter += 10


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

    plt.rcParams['image.cmap'] = 'viridis'
    plt.rcParams["font.family"] = "Times New Roman"

    pool = Pool(process_count)
    labels = ["Static AE", "Random AE",
              "Full History AE", "Full History DAE",
              "Latest Set AE", "Latest Set DAE",
              "Novelty Archive AE", "Novelty Archive DAE"]
    colors = ['black', 'red', 'blue', 'green', 'brown', 'orange', 'purple', 'cyan']
    keys = ["Node Complexity", "Connection Complexity", "Archive Size", "Best Novelty", "Mean Novelty"]

    """pop = load_training_set(labels[2])
    encoder, decoder = load_autoencoder(labels[2], 9)

    for i in range(30, len(pop[-1])):
        lattice = pop[-1][i]
        compressed = encoder.predict(lattice[None])[0]
        reco = decoder.predict(compressed[None])[0]
        auto_encoder_plot(lattice, compressed, reco, calculate_error(lattice, reco))"""

    # for key in keys:
    #     grid_plot(labels, neat_metric, key, key)
    # grid_plot(labels, pca_graph, "Diversity", args=(pca_population(labels)))
    # grid_plot(labels, lattice_diversity, "Diversity", shareAxes=True)
    grid_plot(labels, reconstruction_accuracy, "Reconstruction Error", args=test_population(labels), shareAxes=False)
    # expressive_analysis(labels)
    novelty_spectrum(labels)

    pool.close()
    pool.join()
