from Evaluation.DataLoading import load_autoencoder, load_seed_set, load_training_set, load_seed_pops, load_populations, \
    block_buildings, medieval_population
from Evaluation.EvalutationConfig import *


def reconstruction_accuracy(experiment, pool, args):
    means = []
    cis = []
    for phase in range(-1, phases_to_evaluate):
        print("Loading Autoencoder from Phase {}".format(phase))
        encoder, decoder = load_autoencoder(experiment, phase)
        errors = []

        if args[0] == "Blocks" or args[0] == "Medieval":
            target = [args[1]]
        elif args[0] is not None:
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


def reconstruction_accuracy_new(experiment, population):
    means = []
    cis = []

    for phase in range(-1, phases_to_evaluate):
        print("Loading Autoencoder from Phase {}".format(phase))
        encoder, decoder = load_autoencoder(experiment, phase)
        errors = []

        for building in population[phase]:
            compressed = encoder.predict(building[None])[0]
            reconstructed = decoder.predict(compressed[None])[0]
            errors.append(calculate_error(building, reconstructed))

        means.append(np.mean(errors))
        cis.append(np.std(errors) / np.sqrt(len(errors)) * 1.96)

        if experiment == "Static AE":
            means *= 10
            cis *= 10
            break

    return range(len(means)), np.asarray(means), np.asarray(cis)


def batch_evaluation(experiments):
    batches = ["final", "seed", "medieval", "blocks"]
    results = {}

    for batch in batches:
        print("Processing Batch: ", batch)
        # Load the target population for the batch

        if batch == "medieval":
            population = [medieval_population(True)] * 10
        elif batch == "blocks":
            population = [block_buildings()] * 10
        elif batch == "seed":
            seed_set = load_seed_pops()
            population = [seed_set[0]] * 10

        for experiment in experiments:

            if batch == "final":
                population = load_populations(experiment)
                new = []
                for i in range(10):
                    new_add = []
                    for j in range(10):
                        new_add += population[i][j]
                    new.append(new_add)
                population = new

            print("Processing Experiment: ", experiment)
            # Calculate the reconstruction accuracy for each experiment
            phases, means, cis = reconstruction_accuracy_new(experiment, population)

            # Store the results in a dictionary
            results[experiment + "_" + batch] = [means.tolist(), cis.tolist()]

    # Save results using numpy
    np.save("batch_evaluation_results.npy", results)

    return results


if __name__ == "__main__":
    batch_evaluation(labels)
