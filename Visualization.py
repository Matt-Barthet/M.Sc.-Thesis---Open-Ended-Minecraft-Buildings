import time
import matplotlib.pyplot as plt
import numpy as np

from Delenox_Config import current_run, runs_per_phase


def plot_statistics(metrics, keys, current_run):

    for key in keys:
        plt.figure()
        plt.title("{} vs Generation over {:d} Runs.".format(key, runs_per_phase))
        plt.xlabel("Generation")
        plt.ylabel(key)
        plt.errorbar(x=range(len(metrics[key])),
                     y=metrics[key],
                     yerr=metrics[key + ' confidence'],
                     fmt='-o')
        plt.grid()
        plt.savefig("./Delenox_Experiment_Data/Run{}/Stats_{}.png".format(current_run, key))


def voxel_plot(lattice, title, filename=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(lattice, edgecolor="k", facecolors=get_color_map(lattice))
    ax.set_title(title)
    if filename is not None:
        plt.savefig(
            "./Delenox_Experiment_Data/Run" + str(current_run) + "/Lattice_" + str(time.time()) + ".png")
        plt.cla()
        plt.clf()
    else:
        plt.show()


def novelty_voxel_plot(lattices, generation):
    fig = plt.figure(figsize=(15, 6))
    fig.tight_layout(pad=3)
    titles = ["Least Novel", "Mid-Level Novel", "Most Novel"]
    fig.suptitle("Range of Buildings - Generation: " + str(generation), fontsize=14)
    for number in range(1, len(lattices) + 1):
        ax = fig.add_subplot(1, 3, number, projection='3d')
        ax.set_title(titles[number - 1])
        ax = fig.gca(projection='3d')
        ax.voxels(lattices[number - 1], edgecolor="k", facecolors=get_color_map(lattices[number - 1]))
    plt.savefig("./Delenox_Experiment_Data/Run{}/Lattices_{:f}.png".format(current_run, time.time()))
    # np.save("./Generated_Database/Least_Novel_" + str(time.time()) + ".npy", lattices[0])
    # np.save("./Generated_Database/Mid_Novel_" + str(time.time()) + ".npy", lattices[1])
    # np.save("./Generated_Database/Most_Novel_" + str(time.time()) + ".npy", lattices[2])
    # plt.show()


def expressive_graph(x, y, title, x_label, y_label):
    histogram, x_edges, y_edges = np.histogram2d(x=x, y=y)
    fig = plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    pops = plt.imshow(histogram, interpolation='nearest', origin='low', aspect='auto',
                      extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap=plt.cm.get_cmap("gray"))
    fig.colorbar(pops, label="Building Frequency")
    plt.savefig("./Delenox_Experiment_Data/Run"+str(current_run)+"/Clustering_"+str(time.time())+".png")
    plt.show()


def auto_encoder_plot(example, code, reconstruction, error, title=""):

    fig = plt.figure(figsize=plt.figaspect(2.25))
    fig.suptitle(title)
    ax = fig.add_subplot(3, 1, 1, projection='3d')
    ax.set_title("Original Lattice")
    ax = fig.gca(projection='3d')
    ax.voxels(example, edgecolor="k", facecolors=get_color_map(example))

    ax = fig.add_subplot(3, 1, 2)
    im = ax.imshow(np.array([code] * 15))
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title("Compressed Representation - " + str(len(code)) + " bits")

    ax = fig.add_subplot(3, 1, 3, projection='3d')
    ax.set_title("Reconstructed Lattice - Error: " + str(error) + "%")
    ax = fig.gca(projection='3d')
    ax.voxels(reconstruction, edgecolor="k", facecolors=get_color_map(reconstruction))
    plt.savefig("./Delenox_Experiment_Data/Run"+str(current_run)+"/Autoencoder_"+str(time.time())+".png")
    # plt.show()


def plot_fitness(averages, stdev, generation_count, title, label=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.errorbar(generation_count, averages,
                yerr=stdev,
                fmt='-o',
                label=label)
    if label is not None:
        plt.legend()
    plt.show()


def get_color_map(lattice):
    color = np.empty(lattice.shape, dtype=object)
    for i in range(0, lattice.shape[0]):
        for j in range(0, lattice.shape[1]):
            for k in range(0, lattice.shape[2]):
                if lattice[i][j][k] == 1:
                    color[i][j][k] = 'blue'
                elif lattice[i][j][k] == 2:
                    color[i][j][k] = 'red'
                elif lattice[i][j][k] == 3:
                    color[i][j][k] = 'green'
                elif lattice[i][j][k] == 4:
                    color[i][j][k] = 'yellow'
                elif lattice[i][j][k] == 5:
                    color[i][j][k] = 'pink'
                elif lattice[i][j][k] == 6:
                    color[i][j][k] = 'cyan'
    return color


def visualize_training(history):
    """

    :param history:
    """
    plt.figure()
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Categorical Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation_Set'], loc='upper left')
    plt.savefig("./Delenox_Experiment_Data/Run{}/Training_History.png".format(current_run))

