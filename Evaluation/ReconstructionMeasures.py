from Evaluation.DataLoading import load_autoencoder, load_seed_set, load_training_set, load_seed_pops, load_populations
from Evaluation.EvalutationConfig import *


def reconstruction_accuracy(experiment, pool, args):
    means = []
    cis = []
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
    return range(len(means)), np.asarray(means), np.asarray(cis)


def reconstruct_final_phase(experiment, pool, args):

    print("Loading Autoencoder from Phase 10")
    encoder, decoder = load_autoencoder(experiment, 9)
    target = []

    if args[0][0] == "Blocks" or args[0][0] == "Medieval":
        target = args[0][1]
    elif args[0][0] == "Pops":
        for i in range(10):
            target += list(load_populations(experiment)[-1][i])
    elif args[0][0] == "Seed":
        for i in range(10):
            target += list(args[0][1][i])

    errors = []
    for lattice in target:
        compressed = encoder.predict(lattice[None])[0]
        reconstructed = decoder.predict(compressed[None])[0]
        errors.append(calculate_error(lattice, reconstructed))
    mean = [np.mean(errors)]
    ci = [np.std(errors) / np.sqrt(len(errors)) * 1.96]
    return range(1), np.asarray(mean), np.asarray(ci)
