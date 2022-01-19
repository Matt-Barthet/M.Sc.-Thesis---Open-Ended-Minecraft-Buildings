import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
from scipy.stats import pearsonr

from Evaluation.EvalutationConfig import *
from Evaluation.DataLoading import load_seed_pops, load_populations, load_autoencoder


def subplots_test(labels):

    fig = plt.figure(figsize=(12, 6))

    experiment = 0

    for label in labels:

        if label != "Seed":
            pops = load_populations(label)[-1]
        else:
            pops = load_seed_pops()

        lattice_dict = {}
        counter = 0
        for population in pops:
            for lattice in population:
                lattice_dict.update({counter: lattice})
                counter += 1

        diversities = np.load("./Results/Nov_Plot_{}.npy".format(label), allow_pickle=True).item()
        sorted_keys = [k for k, _ in sorted(diversities.items(), key=lambda item: item[1])]

        ax = fig.add_subplot(3, 6, experiment + 1, projection='3d')
        ax.set_title(label)
        lattice = convert_to_integer(lattice_dict[sorted_keys[-1]])
        ax.voxels(lattice, edgecolor="k", facecolors=get_color_map(lattice, 'blue'))
        ax.set_axis_off()

        ax = fig.add_subplot(3, 6, experiment + 7, projection='3d')
        lattice = convert_to_integer(lattice_dict[sorted_keys[int(len(sorted_keys)/2)]])
        ax.voxels(lattice, edgecolor="k", facecolors=get_color_map(lattice, 'blue'))
        ax.set_axis_off()

        ax = fig.add_subplot(3, 6, experiment + 13, projection='3d')
        lattice = convert_to_integer(lattice_dict[sorted_keys[0]])
        ax.voxels(lattice, edgecolor="k", facecolors=get_color_map(lattice, 'blue'))
        ax.set_axis_off()

        experiment += 1

    plt.tight_layout()
    plt.show()
        # voxel_plot(convert_to_integer(lattice_dict[sorted_keys[1]]),
            #       "Diversity: {}".format(np.round(diversities[sorted_keys[1]], 2)))
        # voxel_plot(convert_to_integer(lattice_dict[sorted_keys[int(len(sorted_keys) / 2)]]),
           #        "Diversity: {}".format(np.round(diversities[sorted_keys[int(len(sorted_keys) / 2)]], 2)))
        # voxel_plot(convert_to_integer(lattice_dict[sorted_keys[-1]]),
          #         "Diversity: {}".format(np.round(diversities[sorted_keys[-1]], 2)))





def plot_test(label, pool):

    # seed = [softmax(np.asarray(lattice, dtype='float').ravel()) for lattice in load_seed_pops()[0]]
    encoder, _ = load_autoencoder(label, 9)
    if label != "Seed":
        pops = load_populations(label)[-1]
    else:
        pops = load_seed_pops()

    lattices = []
    lattice_dict = {}

    counter = 0
    for population in pops:
        for lattice in population:
            lattice_dict.update({counter: lattice})
            counter += 1

    try:
        diversities = np.load("./Results/Nov_Plot_{}.npy".format(label), allow_pickle=True).item()
    except FileNotFoundError:

        counter = 0
        for population in pops:
            this_pop = {}
            for lattice in population:
                # this_pop.update({counter: softmax(np.asarray(lattice, dtype='float').ravel())})
                this_pop.update({counter: encoder.predict(np.asarray(lattice, dtype='float')[None])[0]})
                counter += 1
            lattices.append(this_pop)

        diversities = {}
        for population in lattices:
            print("Working...")
            jobs = [pool.apply_async(novelty_search2, (identifier, lattice, population)) for identifier, lattice in population.items()]
            for result in jobs:
                identifier, diversity = result.get()
                diversities.update({identifier: diversity})
        np.save("./Results/Nov_Plot_{}.npy".format(label), diversities)

    sorted_keys = [k for k, _ in sorted(diversities.items(), key=lambda item: item[1])]

    voxel_plot(convert_to_integer(lattice_dict[sorted_keys[1]]),
               "Diversity: {}".format(np.round(diversities[sorted_keys[1]], 2)))
    voxel_plot(convert_to_integer(lattice_dict[sorted_keys[int(len(sorted_keys) / 2)]]),
               "Diversity: {}".format(np.round(diversities[sorted_keys[int(len(sorted_keys) / 2)]], 2)))
    voxel_plot(convert_to_integer(lattice_dict[sorted_keys[-1]]),
               "Diversity: {}".format(np.round(diversities[sorted_keys[-1]], 2)))

    # for i in range(1, len(sorted_keys)):
        # voxel_plot(convert_to_integer(lattice_dict[sorted_keys[-i]]), "Diversity: {}".format(np.round(diversities[sorted_keys[-i]], 2)))


def novelty_search2(identifier, genome, compressed_population):
    distances = []
    for neighbour in list(compressed_population.values()):
        distance = 0
        for element in range(len(neighbour)):
            distance += np.square(genome[element] - neighbour[element])
        distances.append(np.sqrt(distance))
    distances = np.sort(distances)
    return identifier, np.round(np.mean(distances[1:6]), 2)


def vector_entropy(identifier, lattice, population):
    entropies = []
    for neighbour in population:
        entropies.append(entropy(lattice, neighbour))
    return identifier, np.mean(np.sort(entropies)[1:6])


def novelty_search(genome, compressed_population):
    distances = []
    for neighbour in compressed_population.values():
        distance = 0
        if (genome == neighbour).all():
            continue
        for element in range(len(neighbour)):
            distance += np.square(genome[element] - neighbour[element])
        distances.append(np.sqrt(distance))
    distances = np.sort(distances)
    return np.round(np.average(distances), 2)


def diversity_from_target(experiment, pool, args=None):

    experiment_populations = [load_seed_pops()] + load_populations(experiment)
    experiment_diversity = []
    targets = None

    if args[0] is None:
        pass

    elif isinstance(args[0], int):
        experiment_populations = [experiment_populations[args[0]]]

    # If the target population given is already sorted into individual populations.
    elif len(args[0]) == runs_per_phase:
        targets = [[softmax(np.asarray(lattice, dtype='float')).ravel() for lattice in args[0][population]] for population in range(runs_per_phase)]

    # Otherwise if a single population is given, duplicate it the required number of times.
    else:
        targets = [[softmax(np.asarray(lattice, dtype='float')).ravel() for lattice in args[0]]] * (runs_per_phase)

    for phase in range(len(experiment_populations)):
        phase_diversities = []
        for population in range(len(experiment_populations[phase])):
            print("Starting Experiment {} - Phase {} - Population {}".format(experiment, phase, population))
            lattices = [softmax(np.asarray(lattice, dtype='float')).ravel() for lattice in experiment_populations[phase][population]]

            if args[0] is None or targets is None:
                target_pop = lattices
            else:
                target_pop = targets[population]

            results = [pool.apply_async(vector_entropy, (lattice, target_pop)) for lattice in lattices]
            phase_diversities.append(np.mean([result.get() for result in results]))
        experiment_diversity.append(phase_diversities)

    means = np.mean(experiment_diversity, axis=1)
    ci = np.std(experiment_diversity, axis=1) / np.sqrt(10) * 1.96
    return range(len(means)), means, ci


def func(lattice, vector, compressed, lattices):
    distances = {}
    for identifier, neighbour in compressed.items():
        distance = 0
        if (vector == neighbour).all():
            continue
        for element in range(len(neighbour)):
            distance += np.square(vector[element] - neighbour[element])
        distances.update({identifier: np.sqrt(distance)})
    distances = {identifier: distances for identifier, distances in distances.items()}
    sorted_keys = [k for k, _ in sorted(distances.items(), key=lambda item: item[1])]
    correlations = []
    for i in range(4):
        novelty = []
        entropies = []
        for key in sorted_keys[int(i * len(sorted_keys) / 4): int((i+1) * len(sorted_keys) / 4)]:
            novelty.append(distances[key])
            entropies.append(entropy(lattice, lattices[key]))
        correlations.append(pearsonr(np.asarray(novelty), np.asarray(entropies))[0])
    return correlations


def diversityNovelty(lattice, vector, compressed, lattices):
    distances = {}
    for identifier, neighbour in compressed.items():
        distance = 0
        if (vector == neighbour).all():
            continue
        for element in range(len(neighbour)):
            distance += np.square(vector[element] - neighbour[element])
        distances.update({identifier: np.sqrt(distance)})
    distances = {identifier: distances for identifier, distances in distances.items()}
    sorted_keys = [k for k, _ in sorted(distances.items(), key=lambda item: item[1])]
    novelty = []
    entropies = []
    for key in sorted_keys:
        novelty.append(distances[key])
        entropies.append(entropy(lattice, lattices[key]))
    return novelty, entropies


def diversity_correlation(experiments, pool, args=None):

    results_dict = {}
    try:
        results_dict = np.load("./Results/Diversity_Correlation.npy", allow_pickle=True).item()
    except FileNotFoundError:
        pass

    for experiment in experiments:
        try:
            novelties = results_dict[experiment][0]
            entropies = results_dict[experiment][1]
        except KeyError:
            encoder, _ = load_autoencoder(experiment, 9)
            experiment_populations = load_populations(experiment)[-1]
            novelties = []
            entropies = []
            for population in range(len(experiment_populations)):
                print("Starting Experiment {} - Phase {} - Population {}".format(experiment, 9, population))
                compressed = {identifier: encoder.predict(lattice[None])[0] for identifier, lattice in
                              enumerate(experiment_populations[population])}
                lattices = {identifier: softmax(np.asarray(lattice, dtype='float')).ravel() for identifier, lattice in
                            enumerate(experiment_populations[population])}
                results = [pool.apply_async(diversityNovelty, (lattice, vector, compressed, lattices)) for
                           lattice, vector in zip(lattices.values(), compressed.values())]
                for result in results:
                    novelty, entropy = result.get()
                    novelties += novelty
                    entropies += entropy

        results_dict.update({experiment: [novelties, entropies]})
        np.save("./Results/Diversity_Correlation.npy", results_dict)
        heatmap, xedges, yedges = np.histogram2d(novelties, entropies, bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.figure()
        plt.imshow(heatmap, origin='lower', extent=extent, aspect=xedges[-1]/yedges[-1])
        plt.title(experiment)
        plt.ylabel("Voxel KL-Divergence")
        plt.xlabel("Novelty (Latent Space)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("./Figures/{}-Correlation.png".format(experiment))
        plt.show()


