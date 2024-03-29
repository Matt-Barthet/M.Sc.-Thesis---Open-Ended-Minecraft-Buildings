from Generator.Autoencoder import auto_encoder_3d, create_auto_encoder, add_noise_parallel
from Generator.Delenox_Config import *
from Generator.NeatGenerator import NeatGenerator

if __name__ == '__main__':

    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    # To run an experiment - specify the following four parameters and an experiment title and run the main script
    static = False
    noisy = False
    full_history = True
    random_ae = False
    train_on_archive = False
    experiment = "FH-AE"

    if not os.path.exists('Results/{}'.format(experiment)):
        os.makedirs('Results/{}'.format(experiment))
    else:
        print("Experiment output directory already exists, resuming experiment...")

    for phase in range(number_of_phases):

        if not os.path.exists('Results/{}/Phase{:d}'.format(experiment, phase)):
            os.makedirs('Results/{}/Phase{:d}'.format(experiment, phase))
            neat_metrics = {'Experiment': experiment, 'Mean Novelty': [], 'Best Novelty': [], 'Node Complexity': [],
                            'Infeasible Size': [], 'Connection Complexity': [], 'Archive Size': [],
                            'Species Count': [], 'Mean Genetic Diversity': [],
                            'Minimum Species Size': [], 'Maximum Species Size': [], 'Mean Species Size': []}
            training_population = []
        else:
            try:
                neat_metrics = np.load("./Results/{}/Phase{:d}/Metrics.npz".format(experiment, phase),
                                       allow_pickle=True)['arr_0'].item()
                training_population = list(
                    np.load("./Results/{}/Phase{:d}/Training_Set.npz".format(experiment, phase),
                            allow_pickle=True)['arr_0'])
            except FileNotFoundError:
                neat_metrics = {'Experiment': experiment, 'Mean Novelty': [], 'Best Novelty': [],
                                'Node Complexity': [],
                                'Infeasible Size': [], 'Connection Complexity': [], 'Archive Size': [],
                                'Species Count': [], 'Mean Genetic Diversity': [],
                                'Minimum Species Size': [], 'Maximum Species Size': [], 'Mean Species Size': []}
                training_population = []

        neat_generators = []
        if phase == 0:
            for runs in range(runs_per_phase):
                with open("Results/Seed/Neat_Population_{:d}.pkl".format(runs), "rb") as file:
                    neat_generators.append(pickle.load(file))
        else:
            for runs in range(runs_per_phase):
                with bz2.BZ2File(
                        "Results/{}/Phase{:d}/Neat_Population_{:d}.bz2".format(experiment, phase - 1,
                                                                                               runs), "rb") as file:
                    neat_generators.append(pickle.load(file))

        for number in range(len(neat_generators)):

            if os.path.exists(
                    'Results/{}/Phase{:d}/Neat_Population_{:d}.bz2'.format(experiment, phase, number)):
                continue

            generator, best_fit, metrics = neat_generators[number].run_neat(phase, experiment, static, noise=noisy,
                                                                            train_on_archive=train_on_archive)
            training_population += list(best_fit)

            # Save the neat populations to pickle files in the current phase folder
            with bz2.BZ2File(
                    "./Results/{}/Phase{:d}/Neat_Population_{:d}.bz2".format(experiment, phase, number),
                    'wb') as compressed_file:
                pickle.dump(generator, compressed_file)

            # Save the latest additions to the novel population to a numpy file
            np.savez_compressed("./Results/{}/Phase{:d}/Training_Set.npz".format(experiment, phase),
                                np.asarray(training_population))

            # Update the metrics dictionary with this phase' results
            for key in metrics.keys():
                try:
                    neat_metrics[key].append(metrics[key])
                except KeyError:
                    neat_metrics.update({key: metrics[key]})
                except AttributeError:
                    pass

            # Save the latest metrics to a numpy file for later extraction
            np.savez_compressed("./Results/{}/Phase{:d}/Metrics.npz".format(experiment, phase),
                                neat_metrics)

        if not static and not os.path.exists(
                'Results/{}/Phase{:d}/encoder.json'.format(experiment, phase)):
            training_history = []

            if not full_history:
                start = phase
            else:
                start = 0

            for rewind in range(start, phase + 1):
                training_history += list(
                    np.load("./Results/{}/Phase{:d}/Training_Set.npz".format(experiment, rewind))[
                        'arr_0'])

            if noisy:
                noisy_population = add_noise_parallel(np.asarray(training_history))
            else:
                noisy_population = None

            if random_ae:
                training_pop = None
            else:
                training_pop = np.asarray(training_history)

            ae, encoder, decoder = create_auto_encoder(model_type=auto_encoder_3d,
                                                       phase=phase,
                                                       experiment=experiment,
                                                       population=training_pop,
                                                       noisy=noisy_population
                                                       )
        plt.close('all')
