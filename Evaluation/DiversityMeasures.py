import numpy as np
from scipy.stats import pearsonr

from Evaluation.EvalutationConfig import *
from Evaluation.DataLoading import load_seed_pops, load_populations, load_autoencoder


def vector_entropy(vector1, population):
    entropies = []
    for neighbour in population:
        entropies.append(entropy(vector1, neighbour))
    return np.mean(np.sort(entropies))


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

    novelty = []
    entropies = []
    for key in sorted_keys:
        novelty.append(distances[key])
        entropies.append(entropy(lattice, lattices[key]))

    return pearsonr(np.asarray(novelty), np.asarray(entropies))[0]


def diversity_correlation(experiments, pool, args=None):

    for experiment in experiments:
        for target_pop in experiments:

            if target_pop == experiment:
                continue

            encoder, _ = load_autoencoder(experiment, 9)
            experiment_populations = [load_populations(target_pop)[-1]]
            experiment_diversity = []

            for phase in range(len(experiment_populations)):
                correlations = []
                for population in range(len(experiment_populations[phase])):
                    print("Starting Experiment {} - Phase {} - Population {}".format(experiment, phase, population))
                    compressed = {identifier: encoder.predict(lattice[None])[0] for identifier, lattice in enumerate(experiment_populations[phase][population])}
                    lattices = {identifier: softmax(np.asarray(lattice, dtype='float')).ravel() for identifier, lattice in enumerate(experiment_populations[phase][population])}
                    results = [pool.apply_async(func, (lattice, vector, compressed, lattices)) for lattice, vector in zip(lattices.values(), compressed.values())]
                    correlations = [result.get() for result in results]
                experiment_diversity.append(correlations)

            means = np.mean(experiment_diversity, axis=1)
            ci = np.std(experiment_diversity, axis=1) / np.sqrt(10) * 1.96
            print("Model: {} - Population: {} - {} ± {}".format(experiment, target_pop, means[0], ci[0]))

