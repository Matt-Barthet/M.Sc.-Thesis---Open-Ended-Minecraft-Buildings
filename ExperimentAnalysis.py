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
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

from Autoencoder import load_model, convert_to_integer, test_accuracy, create_auto_encoder, auto_encoder_3d
from Constraints import apply_constraints
from Delenox_Config import value_range
from Visualization import voxel_plot

plt.style.use('seaborn')


def pca_buildings(populations, phases):

    converted_population = []
    for population in populations:
        converted_phases = []
        for phase in population:
            converted_lattices = []
            for lattice in phase:
                converted_lattices.append(convert_to_integer(lattice).ravel())
            converted_phases.append(converted_lattices)
        converted_population.append(converted_phases)

    fit = []
    for population in converted_population:
        for phase in population:
            for lattice in phase:
                fit.append(lattice)

    pca = PCA(n_components=2)
    pca.fit(fit)
    fit.clear()

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


def accuracy_plot(populations, phases, models):
    df = pd.DataFrame({'Phase': [], 'Reconstruction Error': [], 'Experiment': []})

    for population in range(len(models)):

        for phase in phases:
            print("Starting new phase")

            if population != 0:
                encoder = load_model("{}Phase{}/encoder".format(models[population], 6))
                decoder = load_model("{}Phase{}/decoder".format(models[population], 6))
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


def lattice_diversity(populations):
    diversities = []
    df = pd.DataFrame({'Phase': [], 'Entropy': [], 'Experiment': []})

    for population in range(len(populations)):
        for phase in populations[population]:
            print("Starting Phase")

            for lattice in phase:
                diversity = 0

                for other in phase:
                    for (x, y, z) in value_range:
                        diversity += entropy(lattice[x][y][z], other[x][y][z])

                diversities.append(diversity / 8000)

            tmp = pd.DataFrame({'Phase': [phase + 1],
                                'Entropy': [np.round(np.mean(diversities), 2)],
                                'Experiment': [labels[population]]})

            df = pd.concat([df, tmp], axis=0)

    ax = sns.catplot(x='Phase', y='Entropy', hue='Experiment', data=df, kind='bar')
    ax.fig.suptitle('Entropy of Training Sets')
    plt.show()


def novel_diversity(populations):

    eucl_df = pd.DataFrame({'Phase': [], 'Eucl. Distance': [], 'Experiment': []})
    entropy_df = pd.DataFrame({'Phase': [], 'Entropy': [], 'Experiment': []})

    _, encoder, decoder = create_auto_encoder(auto_encoder_3d,
                                              noisy=None,
                                              phase=0,
                                              save=False)
    pool = Pool(11)

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
    diversities = []
    for neighbour in population:
        if not np.array_equal(vector1, neighbour):
            diversities.append(entropy(vector1, neighbour))
    return np.mean(diversities)


def plot_metric(metric_list, labels, colors, keys):

    for key in keys:

        plt.figure()
        plt.title("{} vs Generation over {:d} Runs.".format(key, 7))
        plt.xlabel("Generation")
        plt.ylabel(key)

        for counter in range(len(metric_list)):

            metric = metric_list[counter].item().get(key)
            generations = range(len(metric))

            # Plotting the mean of given metric over generations
            plt.errorbar(x=generations,
                         y=np.mean(metric[generations], axis=-1),
                         fmt='-',
                         label=labels[counter],
                         alpha=1,
                         color=colors[counter])

            # Filling the deviation from the mean in a translucent color.
            plt.fill_between(x=generations,
                             y1=np.mean(metric[generations], axis=-1) + np.std(metric, axis=-1)[generations],
                             y2=np.mean(metric[generations], axis=-1) - np.std(metric, axis=1)[generations],
                             color=colors[counter],
                             alpha=0.25)

        plt.grid()
        legend = plt.legend(frameon=True)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
    plt.show()


def fix_bugged_population(population):
    fixed = [population[0][-subset_size:]]
    for i in range(6):
        offset = []
        for j in range(10):
            offset += (list(range(j * 600 + i * 100, j * 600 + i * 100 + 100)))
        fixed.append(population[-1][offset][-subset_size:])
    return fixed


"""
def cluster_analysis(population, metrics, title, axis_labels, config):
    :param population:
    :param metrics:
    :param title:
    :param axis_labels:
    :param config:
    :return:
    clustering = KMedoids(n_clusters=5)
    data = np.asarray(list(zip(list(metrics[0].values()), list(metrics[1].values()))))
    data_dict = {k: [d[k] for d in metrics] for k in metrics[0].keys()}
    clustering.fit(data)
    clusters = clustering.predict(data)
    medoids = clustering.cluster_centers_

    for medoid in medoids:
        for genome, metrics in data_dict.items():
            if list(medoid) == list(metrics):
                medoid_lattice = generate_lattice(population[genome], config, False)[0][0]
                voxel_plot(convert_to_integer(medoid_lattice), "Medoid at " + str(list(medoid)), "")
                break

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=clusters, s=50, cmap='viridis')
    plt.scatter(medoids[:, 0], medoids[:, 1], c='black', s=200, alpha=0.5)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.savefig("./Delenox_Experiment_Data/Run" + str(current_run) + "/Clustering_" + str(time.time()) + ".png")
    plt.show()
"""


if __name__ == '__main__':

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    subset_size = 1000

    labels = [
        "Static DAE",
        "Random AE",
        "Latest Set DAE",
        "Full History DAE",
        "Full History AE",
        "Novelty Archive AE"
    ]

    colors = [
        'black',
        'red',
        'blue',
        'green',
        'brown',
        'yellow'
    ]

    keys = [
        "Node Complexity",
        "Connection Complexity",
        "Archive Size",
        "Best Novelty",
        "Mean Novelty"
    ]

    directories = [
        './Static Denoising AE - Clearing Archive/',
        './Random AE - Clearing Archive/',
        './Retrain Denoising AE (Latest Batch) - Clearing Archive/',
        './Retrain Denoising AE (Full History) - Clearing Archive/',
        './Retrain Vanilla AE (Full History) - Clearing Archive/',
        './Retrain AE (Full Archive History) - Clearing Archive/'
    ]

    training_sets = [
        [np.load("./Static Denoising AE - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)],
        [np.load("./Random AE - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)],
        [np.load("./Retrain Denoising AE (Latest Batch) - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)],
        fix_bugged_population([np.load("./Retrain Denoising AE (Full History) - Clearing Archive/Phase{}/Training_Set.npy".format(i)) for i in range(7)]),
        [np.load("./Retrain Vanilla AE (Full History) - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)],
        [np.load("./Retrain AE (Full Archive History) - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)]
    ]

    metrics = [
        np.load("./Static Denoising AE - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True),
        np.load("./Random AE - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True),
        np.load("./Retrain Denoising AE (Latest Batch) - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True),
        np.load("./Retrain Denoising AE (Full History) - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True),
        np.load("./Retrain Vanilla AE (Full History) - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True),
        np.load("./Retrain AE (Full Archive History) - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    ]

    lattices = np.load("Ahousev5_Buildings_Varied.npy")
    for lattice in lattices:
        lattice = apply_constraints(lattice)[1]
        voxel_plot(lattice, "")
        example = to_categorical(lattice, num_classes=5)

    # novel_diversity(training_sets)
    # pca_buildings(training_sets, range(7))
    # accuracy_plot(training_sets, range(7), directories)
    # plot_metric(metrics, labels, colors, keys)
