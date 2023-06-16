import matplotlib.pyplot as plt
import numpy as np
from Generator.Delenox_Config import runs_per_phase, lattice_dimensions
from mpl_toolkits.mplot3d import axes3d, Axes3D


def plot_statistics(values, confidence, key, phase):
    plt.figure()
    plt.title("{} vs Generation using {:d} Populations.".format(key, runs_per_phase))
    plt.xlabel("Generation")
    plt.ylabel(key)
    plt.errorbar(x=range(len(values)),
                 y=values,
                 yerr=confidence,
                 fmt='-o')
    plt.grid()
    plt.savefig("./Results/Phase{}/{}_Stats.png".format(phase, key))


def voxel_plot(lattice, title, filename=None, color_one='blue'):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(lattice, edgecolor="k", facecolors=get_color_map(lattice, color_one))
    ax.set_title(title)
    ax.set_axis_off()
    fig.add_axes(ax)
    if filename is not None:
        plt.savefig(filename)
        plt.cla()
        plt.clf()
    else:
        plt.show()


def novelty_voxel_plot(lattices, generation, population_id, phase, experiment):
    fig = plt.figure(figsize=(15, 6))
    fig.tight_layout(pad=3)
    titles = ["Least Novel", "Mid-Level Novel", "Most Novel"]
    fig.suptitle("Range of Buildings - Generation: " + str(generation), fontsize=14)
    for number in range(1, len(lattices) + 1):
        ax = fig.add_subplot(1, 3, number, projection='3d')
        ax.set_title(titles[number - 1])
        ax = fig.gca(projection='3d')
        ax.voxels(lattices[number - 1], edgecolor="k", facecolors=get_color_map(lattices[number - 1]))
    plt.savefig("./Results/{}/Phase{}/Lattices_{:d}_Gen{:d}.png".format(experiment, phase, population_id, generation))


def expressive_graph(fig, ax, x, y, title, x_label, y_label):
    if x_label == "Instability" and y_label == "Symmetry":
        histogram, x_edges, y_edges = np.histogram2d(x=x, y=y, bins=[np.linspace(0, 8, 20), np.linspace(0, 1, 20)])
    if x_label == "Surface Area" and y_label == "Instability":
        histogram, x_edges, y_edges = np.histogram2d(x=x, y=y, bins=[np.linspace(0, 28, 20), np.linspace(0, 8, 20)])
    if x_label == "Surface Area" and y_label == "Symmetry":
        histogram, x_edges, y_edges = np.histogram2d(x=x, y=y, bins=[np.linspace(0, 28, 20), np.linspace(0, 1, 20)])

    ax.set_title(title)
    pops = ax.imshow(np.transpose(histogram),
                     interpolation='nearest',
                     origin='lower',
                     aspect='auto',
                     extent=[0, x_edges[-1], 0, y_edges[-1]],
                     cmap=plt.cm.get_cmap("gray"),
                     vmin=0,
                     vmax=30)
    fig.colorbar(pops, ax=ax)


def convert_to_integer(lattice):
    integer_reconstruct = np.zeros(lattice_dimensions)
    for channel in range(20):
        for row in range(20):
            integer_reconstruct[channel][row] = np.argmax(lattice[channel][row], axis=1)
    return integer_reconstruct


def auto_encoder_plot(example, code, reconstruction, error, title=""):
    fig = plt.figure(figsize=plt.figaspect(2.25))
    fig.suptitle(title)
    ax = fig.add_subplot(3, 1, 1, projection='3d')
    ax.set_title("Original Lattice")
    ax = fig.gca(projection='3d')
    ax.voxels(convert_to_integer(example), edgecolor="k", facecolors=get_color_map(convert_to_integer(example)))
    ax = fig.add_subplot(3, 1, 2)
    im = ax.imshow(np.array([code] * 15))
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title("Compressed Representation - " + str(len(code)) + " bits")
    ax = fig.add_subplot(3, 1, 3, projection='3d')
    ax.set_title("Reconstructed Lattice - Error: " + str(error) + "%")
    ax = fig.gca(projection='3d')
    ax.voxels(convert_to_integer(reconstruction), edgecolor="k", facecolors=get_color_map(convert_to_integer(reconstruction)))
    plt.show()


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


def get_color_map(lattice, color_one='blue'):
    color = np.empty(lattice.shape, dtype=object)
    for i in range(0, color.shape[0]):
        for j in range(0, color.shape[1]):
            for k in range(0, color.shape[2]):
                if lattice[i][j][k] == 1:
                    color[i][j][k] = color_one
                elif lattice[i][j][k] == 2:
                    color[i][j][k] = (128/255, 128/255, 128/255)
                elif lattice[i][j][k] == 3:
                    color[i][j][k] = (161/255, 132/255, 79/255)
                elif lattice[i][j][k] == 4:
                    color[i][j][k] = (102/255, 77/255, 46/255)
                elif lattice[i][j][k] == 5:
                    color[i][j][k] = 'pink'
                elif lattice[i][j][k] == 6:
                    color[i][j][k] = 'cyan'
    return color


def visualize_training(history, phase, experiment):
    plt.figure()
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Categorical Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation_Set'], loc='upper left')
    if phase == -1:
        plt.savefig("./Results/Seed/Training_History.png")
    else:
        plt.savefig("./Results/{}/Phase{}/Training_History.png".format(experiment, phase))