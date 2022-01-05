from Evaluation.DataLoading import load_autoencoder, load_seed_set, load_training_set, load_seed_pops, load_populations
from Evaluation.EvalutationConfig import *


def reconstruction_accuracy(experiment, pool, args):
    means = []
    cis = []
    diversity_dict = {}

    for phase in range(-1, phases_to_evaluate):
        print("Loading Autoencoder from Phase {}".format(phase))
        encoder, decoder = load_autoencoder(experiment, phase)
        errors = []

        if args[0] is not None:
            target = args[0]
        elif phase == -1:
            target = load_seed_pops()
        else:
            target = load_populations(experiment)[phase]

        for pop in range(len(target)):
            print("Phase {} - Population {} - Reconstruction Test".format(phase, pop))
            pop_errors = []
            for lattice in target[pop]:
                compressed = encoder.predict(lattice[None])[0]
                reconstructed = decoder.predict(compressed[None])[0]
                pop_errors.append(calculate_error(lattice, reconstructed))
            errors.append(np.mean(pop_errors))

        means.append(np.mean(errors))
        cis.append(np.std(errors) / np.sqrt(len(errors)) * 1.96)

    diversity_dict.update({experiment: [np.asarray(means), np.asarray(cis)]})
    return range(len(means)), np.asarray(means), np.asarray(cis)
