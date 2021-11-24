from Evaluation.EvalutationConfig import *
from Evaluation.DataLoading import load_seed_pops, load_populations


def vector_entropy(vector1, population):
    entropies = []
    for neighbour in population:
        entropies.append(entropy(vector1, neighbour))
    return np.mean(np.sort(entropies))


def novelty(vector, compressed_population):
    distances = []
    for neighbour in compressed_population:
        novelty_score = 0
        for element in range(len(neighbour)):
            novelty_score += np.square(vector[element] - neighbour[element])
        distances.append(np.sqrt(novelty_score))
    distances = np.sort(distances)[1:]
    return np.round(np.average(distances), 2)


def diversity_from_target(experiment, pool, args=None):

    experiment_populations = [load_seed_pops()] + load_populations(experiment)
    experiment_diversity = []

    if args[0] is None:
        pass

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

            if args[0] is None:
                target_pop = lattices
            else:
                target_pop = targets[population]

            results = [pool.apply_async(vector_entropy, (lattice, target_pop)) for lattice in lattices]
            phase_diversities.append(np.mean([result.get() for result in results]))
        experiment_diversity.append(phase_diversities)

    means = np.mean(experiment_diversity, axis=1)
    ci = np.std(experiment_diversity, axis=1) / np.sqrt(10) * 1.96
    return range(len(means)), means, ci
