from Evaluation.DataLoading import load_autoencoder, load_seed_set, load_training_set
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
            target = load_seed_set()
        else:
            target = load_training_set(experiment)[phase]

        for lattice in target:
            if experiment[-3:] == 'DAE':
                compressed = encoder.predict(add_noise(lattice)[None])[0]
            else:
                compressed = encoder.predict(lattice[None])[0]
            reconstructed = decoder.predict(compressed[None])[0]
            errors.append(calculate_error(lattice, reconstructed))

        means.append(np.mean(errors))
        cis.append(np.std(errors) / np.sqrt(len(errors)) * 1.96)

    diversity_dict.update({experiment: [np.asarray(means), np.asarray(cis)]})
    return range(len(means)), np.asarray(means), np.asarray(cis)
