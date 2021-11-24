from Evaluation.DataLoading import load_autoencoder
from Evaluation.EvalutationConfig import *


def reconstruction_accuracy(experiment, pool, args):
    means = []
    cis = []
    diversity_dict = {}
    for phase in range(-1, phases_to_evaluate):
        print("Loading Autoencoder from Phase {}".format(phase))
        encoder, decoder = load_autoencoder(experiment, phase)
        errors = []
        for lattice in args[0]:
            if experiment[-3:] == 'DAE':
                compressed = encoder.predict(add_noise(lattice)[None])[0]
            else:
                compressed = encoder.predict(lattice[None])[0]
            reconstructed = decoder.predict(compressed[None])[0]
            errors.append(calculate_error(lattice, reconstructed))

        if experiment == "Static AE":
            means = np.asarray([np.mean(errors) for _ in range(phases_to_evaluate + 1)])
            cis = np.asarray([np.std(errors) / np.sqrt(len(errors)) * 1.96 for _ in range(phases_to_evaluate + 1)])
            diversity_dict.update({experiment: [means, cis]})
            return range(len(means)), np.asarray(means), np.asarray(cis)
        else:
            means.append(np.mean(errors))
            cis.append(np.std(errors) / np.sqrt(len(errors)) * 1.96)

    diversity_dict.update({experiment: [np.asarray(means), np.asarray(cis)]})
    return range(len(means)), np.asarray(means), np.asarray(cis)
