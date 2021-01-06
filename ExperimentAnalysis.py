import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.decomposition import PCA
from Autoencoder import load_model, convert_to_integer, test_accuracy
from Constraints import change_to_ones
from Delenox_Config import value_range
from Visualization import voxel_plot


def pca_buildings(populations, phases):
    plt.figure()
    plt.title("PCA of Novel Populations")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    eucl_averages = []

    pca = PCA(n_components=2)
    fit = []
    for lattice in populations[-1]:
        fit.append(convert_to_integer(lattice).ravel())
    pca.fit(fit)

    offsets = []
    for model in range(7):
        if model <= 1:
            model = 1
        offset = []
        for i in range(1, 11):
            offset += list(range(i * model * 100 - 100, i * model * 100))
        offsets.append(offset)

    for model in [0, 6]:

        lattices = []

        for lattice in populations[model][offsets[model]]:
            lattices.append(convert_to_integer(lattice).ravel())

        principalComponents = pca.transform(lattices)
        principalDf = pd.DataFrame(data=principalComponents, columns=['1', '2'])

        averages = []
        for vector in principalComponents:
            average = 0
            for other in principalComponents:
                dist = np.linalg.norm(vector - other)
                average = np.mean([average, dist])
            averages.append(average)
        eucl_averages.append(np.mean(averages))

        plt.scatter(principalDf['1'], principalDf['2'], cmap=[model] * len(principalDf), s=10, alpha=0.5, label="Novel Set - Phase {:d}".format(model + 1))

    plt.legend()
    plt.show()

    plt.figure()
    plt.bar(x=range(1, len(eucl_averages) + 1), height=eucl_averages)
    print(eucl_averages)
    plt.xlabel("Exploration Phase")
    plt.ylabel("Average Euclidean Distance (PCA Values)")
    plt.title("Average Distance of PCA Values from each Exploration Phase")
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


if __name__ == '__main__':

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    baseline = [np.load("./Baseline Experiment/Phase{}/Training_Set.npy".format(i))[:100] for i in range(7)]
    latest_batch = [np.load("./Retrain AE (Latest Batch) - Clearing Archive/Phase{}/Training_Set.npy".format(i))[:100] for i in range(7)]
    full_history = [np.load("./Retrain AE (Full History) - Clearing Archive/Phase{}/Training_Set.npy".format(i))[-100:] for i in range(7)]

    # pca_buildings(full_history, range(7))

    # pop = np.load("Training_Set.npy", allow_pickle=True)
    # print(pop.shape)

    #  voxel_plot(change_to_ones(convert_to_integer(pop[899])), "")

    baseline_metrics = np.load("./Baseline Experiment/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    latest_metrics = np.load("./Retrain AE (Latest Batch) - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    full_metrics = np.load("./Retrain AE (Full History) - Clearing Archive/Phase{}/Metrics.npy".format(6), allow_pickle=True)
    key = "Archive Size"
    indices = range(0, 700)
    plt.figure()
    plt.title("{} vs Generation over {:d} Runs.".format(key, 7))
    plt.xlabel("Generation")
    plt.ylabel(key)
    plt.errorbar(x=indices, y=np.mean(latest_metrics.item().get(key)[indices], axis=-1), fmt='-', label="Retrained AE (Latest Set)", alpha=1, color='red')
    plt.fill_between(x=indices, y1=np.mean(latest_metrics.item().get(key)[indices], axis=-1) + np.std(latest_metrics.item().get(key), axis=-1)[indices], y2=np.mean(latest_metrics.item().get(key)[indices], axis=-1) - np.std(baseline_metrics.item().get(key), axis=1), color='red', alpha=0.25)
    plt.errorbar(x=indices, y=np.mean(full_metrics.item().get(key)[indices], axis=-1)[indices], fmt='-', label="Retrained AE (Full History)", alpha=1, color='yellow')
    plt.fill_between(x=indices, y1=np.mean(full_metrics.item().get(key)[indices], axis=-1) + np.std(full_metrics.item().get(key), axis=-1)[indices], y2=np.mean(full_metrics.item().get(key)[indices], axis=-1) - np.std(full_metrics.item().get(key), axis=1), color='yellow', alpha=0.25)
    plt.errorbar(x=indices, y=np.mean(baseline_metrics.item().get(key)[indices], axis=-1), fmt='-', label="Baseline - Static AE", alpha=1, color='blue')
    plt.fill_between(x=indices, y1=np.mean(baseline_metrics.item().get(key)[indices], axis=-1) + np.std(baseline_metrics.item().get(key), axis=-1)[indices], y2=np.mean(baseline_metrics.item().get(key)[indices], axis=-1) - np.std(baseline_metrics.item().get(key), axis=1), color='blue', alpha=0.25)
    plt.grid()
    plt.legend(loc=2)
    plt.show()

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

    """

    """


