import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.decomposition import PCA
from scipy import spatial
from Autoencoder import load_model, convert_to_integer, test_accuracy
from Delenox_Config import value_range


def pca_buildings(populations, phases):
    plt.figure()
    plt.title("PCA of Novel Populations")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    eucl_averages = []

    for model in phases:

        lattices = []
        for lattice in populations[model]:
            lattices.append(convert_to_integer(lattice).ravel())

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(lattices)
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
    plt.xlabel("Exploration Phase")
    plt.ylabel("Average Euclidean Distance (PCA Values)")
    plt.title("Average Distance of PCA Values from each Exploration Phase")
    plt.show()


if __name__ == '__main__':

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    """node_complexity = [[] for _ in range(5)]
    connection_complexity = [[] for _ in range(5)]
    archive_size = [[] for _ in range(5)]

    with open('Raw Output.txt', 'r') as fp:
        while True:
            line = fp.readline()
            if line[1:11] == "Population":
                pop_id = int(line[12])

                node = fp.readline()
                value = 0
                for t in node.split():
                    try:
                        value = float(t)
                    except ValueError:
                        pass

                node_complexity[pop_id].append(value)

                node = fp.readline()
                value = 0
                for t in node.split():
                    try:
                        value = float(t)
                    except ValueError:
                        pass

                connection_complexity[pop_id].append(value)

                node = fp.readline()
                value = 0
                for t in node.split():
                    try:
                        value = float(t)
                    except ValueError:
                        pass

                archive_size[pop_id].append(value)

            if line == "end":
                break

    metrics = np.load("./Delenox_Experiment_Data/Phase4/Metrics.npy", allow_pickle=True)
    print(metrics)
    other = metrics.item().get('Connection Complexity')
    archive_stacked = np.stack(connection_complexity, axis=-1)

    plt.figure()
    plt.title("{} vs Generation over {:d} Iterations.".format("Connection Complexity", 5))
    plt.xlabel("Generation")
    plt.ylabel("Connection Complexity")

    plt.errorbar(x=range(499),
                 y=np.mean(archive_stacked, axis=-1)[:499],
                 yerr=np.std(archive_stacked, axis=-1)[:499],
                 fmt='-o',
                 label="Static AE (Baseline)")

    plt.errorbar(x=range(len(other)),
                 y=np.mean(other, axis=-1),
                 yerr=np.std(other, axis=-1),
                 fmt='-o',
                 label="Evolved AE")

    plt.grid()
    plt.legend()
    plt.show()"""

    populations = [np.load("./Delenox_Experiment_Data/Phase{}/Training_Set.npy".format(i))[-50:] for i in range(5)]
    compressed = [[] for _ in range(5)]

    all = []
    for population in populations:
        lattice_diversities = []
        for lattice in population:
            diversities = []
            for other in population:
                diversity = 0
                for (x, y, z) in value_range:
                    if list(lattice[x][y][z]) != list(other[x][y][z]):
                        diversity += 1
                diversities.append(diversity / 8000)
            lattice_diversities.append(np.average(diversities))
            print("Population Done!")
        all.append(np.mean(lattice_diversities))

    plt.figure()
    plt.bar(x=range(1, 6), height=all)
    plt.xlabel("Lattice Diversity in each Novel Batch")
    plt.ylabel("Diversity")
    plt.title("Population from Phase")
    plt.show()
    pca_buildings(populations, [2, 3, 4])

    """errors = []
    errors0 = []
    encoder0 = load_model("./Delenox_Experiment_Data/Phase{}/encoder".format(0))
    decoder0 = load_model("./Delenox_Experiment_Data/Phase{}/decoder".format(0))

    for model in range(5):
        print("Starting")
        encoder = load_model("./Delenox_Experiment_Data/Phase{}/encoder".format(4))
        decoder = load_model("./Delenox_Experiment_Data/Phase{}/decoder".format(4))
        error = test_accuracy(encoder, decoder, list(populations[model]))
        error0 = test_accuracy(encoder0, decoder0, list(populations[model]))
        errors.append(error)
        errors0.append(error0)

    df = pd.DataFrame({"Evolved AE": errors, "Static AE": errors0}, index=range(1, 6))
    df.plot.bar(rot=0)
    plt.show()
"""
    """for i in range(5):
        print("Compressing Population {:d} with AE from Phase {:d}".format(i + 1, 5))
        for lattice in populations[i]:
            compressed[i].append(encoder.predict(lattice[None])[0])"""
    """ 
    errors = []
    for model in range(len(autoencoders)):
        error = test_accuracy(autoencoders[model]["encoder"], autoencoders[model]["decoder"],
                              rn.sample(list(next_population), 250))
        errors.append(error)
    plt.figure()
    plt.bar(x=range(1, len(errors) + 1), height=errors)
    plt.xlabel("Autoencoder from Phase")
    plt.ylabel("Error %")
    plt.title("Autoencoder Reconstruction Error on Latest Novel Batch")
    plt.savefig("./Delenox_Experiment_Data/Phase{}/Error_Latest.png".format(current_run))"""

    """eucl_averages = []
    for model in range(len(autoencoders)):
        average = 0
        for vector in compressed[model]:
            for other in compressed[model]:
                dist = np.linalg.norm(vector - other)
                average = np.mean([average, dist])
        eucl_averages.append(average)
    plt.figure()
    plt.bar(x=range(1, len(eucl_averages) + 1), height=eucl_averages)
    plt.xlabel("Autoencoder from Phase")
    plt.ylabel("Euclidean Distance")
    plt.title("Average Euclidean Distance (Vectors) on latest Novel Batch")
    plt.savefig("./Delenox_Experiment_Data/Run{}/Eucl_Latest.png".format(current_run))"""

    """inital_compressed = [[] for _ in range(len(autoencoders))]
    eucl_averages = []
    for model in range(len(autoencoders)):
        for lattice in initial_population:
            inital_compressed[model].append(autoencoders[model]['encoder'].predict(lattice[None])[0])
        average = 0
        for vector in inital_compressed[model]:
            for other in compressed[model]:
                dist = np.linalg.norm(vector - other)
                average = np.mean([average, dist])
        eucl_averages.append(average)
    plt.figure()
    plt.bar(x=range(1, len(eucl_averages) + 1), height=eucl_averages)
    plt.xlabel("Autoencoder from Phase")
    plt.ylabel("Euclidean Distance")
    plt.title("Average Euclidean Distance (Vectors) on Initial Population")
    plt.savefig("./Delenox_Experiment_Data/Run{}/Eucl_Initial.png".format(current_run))"""

