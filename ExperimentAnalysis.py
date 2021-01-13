import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.decomposition import PCA

from Autoencoder import load_model, convert_to_integer, test_accuracy
from Delenox_Config import value_range

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


def accuracy_plot(populations):
    errors = []
    errors0 = []
    # encoder0 = load_model("./Retrain AE (Full History) - Clearing Archive/Phase{}/encoder".format(0))
    # decoder0 = load_model("./Retrain AE (Full History) - Clearing Archive/Phase{}/decoder".format(0))

    offsets = []

    for model in range(8):
        if model <= 1:
            model = 1
        offset = []
        for i in range(1, 11):
            offset += list(range(i * model * 100 - 100, i * model * 100))
        offsets.append(offset)

    for model in [1, 6]:
        print("Starting")
        encoder = load_model("./Retrain AE (Latest Batch) - Clearing Archive/Phase{}/encoder".format(model))
        decoder = load_model("./Retrain AE (Latest Batch) - Clearing Archive/Phase{}/decoder".format(model))
        error = test_accuracy(encoder, decoder, list(populations[model + 1]))
        # error0 = test_accuracy(encoder0, decoder0, list(populations[model + 1][offsets[model + 1]]))
        errors.append(error)
        #errors0.append(error0)

    print("Errors Initial: ", errors0)
    print("Errors Final: ", errors)

    df = pd.DataFrame({"Evolved AE": errors, "Static AE": errors0}, index=range(1, 9))
    df.plot.bar(rot=0)
    plt.show()


def lattice_dviersity(lattice, population):
    diversities = []
    for other in population:
        diversity = 0
        for (x, y, z) in value_range:
            if list(lattice[x][y][z]) != list(other[x][y][z]):
                diversity += 1
        diversities.append(diversity / 8000)
    return np.average(diversities)


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

    # pool = Pool(11)
    # results = []

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    labels = ["Static DAE", "Random AE", "Retrained DAE (Latest Set)", "Retrained DAE (Full History)", "Retrained AE (Full History)"]
    colors = ['black', 'red', 'blue', 'green', 'brown']
    keys = ["Node Complexity", "Connection Complexity", "Archive Size", "Best Novelty", "Mean Novelty"]

    subset_size = 1000
    static_dae = [np.load("./Static Denoising AE - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)]
    random_ae = [np.load("./Random AE - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)]
    latest_dae = [np.load("./Retrain Denoising AE (Latest Batch) - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)]
    full_dae = [np.load("./Retrain Denoising AE (Full History) - Clearing Archive/Phase{}/Training_Set.npy".format(i)) for i in range(7)]
    full_dae = fix_bugged_population(full_dae)
    full_ae = [np.load("./Retrain Vanilla AE (Full History) - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-subset_size:] for i in range(7)]
    # pca_buildings([static_dae, random_ae, latest_dae, full_dae, full_ae], range(7))

    static_dae_metrics = np.load("./Static Denoising AE - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    latest_dae_metrics = np.load("./Retrain Denoising AE (Latest Batch) - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    full_dae_metrics = np.load("./Retrain Denoising AE (Full History) - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    full_ae_metrics = np.load("./Retrain Vanilla AE (Full History) - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    random_ae_metrics = np.load("./Random AE - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    metrics = [static_dae_metrics, random_ae_metrics, latest_dae_metrics, full_dae_metrics, full_ae_metrics]
    plot_metric(metrics, labels, colors, keys)

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


