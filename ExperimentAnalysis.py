import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from Autoencoder import load_model, convert_to_integer, test_accuracy, create_auto_encoder, auto_encoder_3d, \
    convert_to_ones
from Delenox_Config import value_range
from Visualization import voxel_plot


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

            tmp = pd.DataFrame({'Phase': [phase + 1],
                                'Average Euclidean Distance': [np.round(np.mean(averages), 2)],
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

    ax = sns.catplot(x='Phase', y='Average Euclidean Distance', hue='Experiment', data=eucl_df, kind='bar')
    ax.fig.suptitle('Average PW Euclidean Distance using PCA')
    plt.show()


def accuracy_plot(populations, phases, models):
    errors = []
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

            error = test_accuracy(encoder, decoder, populations[population][phase])
            errors.append(error)

            tmp = pd.DataFrame({'Phase': [phase + 1],
                                'Reconstruction Error': [np.round(np.mean(error), 2)],
                                'Experiment': [labels[population]]})
            df = pd.concat([df, tmp], axis=0)

    ax = sns.catplot(x='Phase', y='Reconstruction Error', hue='Experiment', data=df, kind='bar')
    ax.fig.suptitle('Average Reconstruction Error using Final Autoencoder')
    plt.show()


def lattice_dviersity(populations):
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

    pool = Pool(16)

    for population in range(len(populations)):
        for phase in range(len(populations[population])):
            print("Starting Phase")

            results = []
            phase_vectors = []
            phase_diversity = []
            phase_entropy = []

            for lattice in populations[population][phase]:
                compressed = encoder.predict(lattice[None])[0]
                compressed = softmax(compressed)
                phase_vectors.append(compressed)

            for vectors in phase_vectors:
                results.append(pool.apply_async(vector_novelty, (vectors, phase_vectors)))

            for result in results:
                phase_diversity.append(result.get())

            results.clear()
            for vectors in phase_vectors:
                results.append(pool.apply_async(vector_entropy, (vectors, phase_vectors)))

            for result in results:
                phase_entropy.append(result.get())

            tmp = pd.DataFrame({'Phase': [phase + 1],
                                'Eucl. Distance': [np.mean(phase_diversity)],
                                'Experiment': [labels[population]]})
            eucl_df = pd.concat([eucl_df, tmp], axis=0)

            tmp = pd.DataFrame({'Phase': [phase + 1],
                                'Entropy': [np.mean(phase_entropy)],
                                'Experiment': [labels[population]]})
            entropy_df = pd.concat([entropy_df, tmp], axis=0)

    ax = sns.catplot(x='Phase', y='Eucl. Distance', hue='Experiment', data=eucl_df, kind='bar')
    ax.fig.suptitle('Average Pairwise Eucl. Distance of Latent Vectors Training Sets')
    plt.show()

    ax = sns.catplot(x='Phase', y='Entropy', hue='Experiment', data=entropy_df, kind='bar')
    ax.fig.suptitle('Average Entropy of Latent Vectors in Training Sets')
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
        plt.legend()
    plt.show()


def fix_bugged_population(population):
    fixed = [population[0][-subset_size:]]
    for i in range(6):
        offset = []
        for j in range(10):
            offset += (list(range(j * 600 + i * 100, j * 600 + i * 100 + 100)))
        fixed.append(population[-1][offset][-subset_size:])
    return fixed


if __name__ == '__main__':

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    labels = ["Static DAE",
              "Random AE",
              "Retrained DAE (Latest Set)",
              "Retrained DAE (Full History)",
              "Retrained AE (Full History)"]

    colors = ['black', 'red', 'blue', 'green', 'brown']
    keys = ["Node Complexity", "Connection Complexity", "Archive Size", "Best Novelty", "Mean Novelty"]

    directories = ['./Static Denoising AE - Clearing Archive/',
                   './Random AE - Clearing Archive/',
                   './Retrain Denoising AE (Latest Batch) - Clearing Archive/',
                   './Retrain Denoising AE (Full History) - Clearing Archive/',
                   './Retrain Vanilla AE (Full History) - Clearing Archive/'
                   ]

    subset_size = 1000
    static_dae = [np.load("./Static Denoising AE - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)]
    random_ae = [np.load("./Random AE - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)]
    latest_dae = [np.load("./Retrain Denoising AE (Latest Batch) - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)]
    full_dae = [np.load("./Retrain Denoising AE (Full History) - Clearing Archive/Phase{}/Training_Set.npy".format(i)) for i in range(7)]
    full_dae = fix_bugged_population(full_dae)
    full_ae = [np.load("./Retrain Vanilla AE (Full History) - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)]

    training_sets = [
        static_dae,
        random_ae,
        latest_dae,
        full_dae,
        full_ae
    ]

    novel_diversity(training_sets)

    """full_dae = np.load("./Retrain Vanilla AE (Full History) - Persistent Archive/Phase4/Population_5.npy", allow_pickle=True)
    full_dae = list(full_dae.item().values())

    counter = 1
    for i in [1, 2, 3, 5, 11]:
        building = np.load("./Building_{:d}.npy".format(i))
        voxel_plot(building, "", "./Buildings/Materials#{:d}".format(counter))
        voxel_plot(convert_to_ones(building), "", "./Buildings/White#{:d}".format(counter), color_one='white')
        voxel_plot(convert_to_ones(building), "", "./Buildings/Blue#{:d}".format(counter), color_one='blue')
        counter+=1"""

    # pca_buildings([static_dae, random_ae, latest_dae, full_dae, full_ae], range(7))

    static_dae_metrics = np.load("./Static Denoising AE - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    latest_dae_metrics = np.load("./Retrain Denoising AE (Latest Batch) - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    full_dae_metrics = np.load("./Retrain Denoising AE (Full History) - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    full_ae_metrics = np.load("./Retrain Vanilla AE (Full History) - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    random_ae_metrics = np.load("./Random AE - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    metrics = [static_dae_metrics, random_ae_metrics, latest_dae_metrics, full_dae_metrics, full_ae_metrics]
    # plot_metric(metrics, labels, colors, keys)

    #accuracy_plot(training_sets, range(7), directories)
    """diversity = []
    for population in full_history:
        pool = Pool(10)
        jobs = []
        pop_diversity = []
        for lattice in population:
            jobs.append(pool.apply_async(lattice_dviersity, (lattice, population)))
        for job in jobs:
            pop_diversity.append(job.get())
        diversity.append(np.mean(pop_diversity))
        print(diversity)"""


