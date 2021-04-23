import matplotlib.pyplot as plt
from Autoencoder import auto_encoder_3d, create_auto_encoder, add_noise, add_noise_parallel
from Delenox_Config import *
from NeatGenerator import NeatGenerator

if __name__ == '__main__':

    # Flag to test a static configuration of DeLeNoX, where no transformation phases take place
    static = False
    noisy = False

    experiment = "Retrained AE - Novelty Archive"
    if not os.path.exists('Delenox_Experiment_Data/{}'.format(experiment)):
        os.makedirs('Delenox_Experiment_Data/{}'.format(experiment))
    else:
        print("Experiment output directory already exists, resuming experiment...")

    # Loop through the phase numbers provided, if the list starts from a value greater than zero, it resumes the experiment from the given phase
    for phase in range(number_of_phases):

        """
        DeLeNoX Iteration Initialization:
        1) Check if there is any output for the current phase of the given experiment.
        2) Create a directory and initialise variables if no output is found.
        3) Load the latest metrics and training set if the phase has already been started.
        """
        if not os.path.exists('Delenox_Experiment_Data/{}/Phase{:d}'.format(experiment, phase)):
            os.makedirs('Delenox_Experiment_Data/{}/Phase{:d}'.format(experiment, phase))
            neat_metrics = {'Experiment': experiment, 'Mean Novelty': [], 'Best Novelty': [], 'Node Complexity': [], 'Connection Complexity': [],
                            'Archive Size': [], 'Species Count': [], 'Infeasible Size': []}
            training_population = []
        else:
            try:
                neat_metrics = np.load("./Delenox_Experiment_Data/{}/Phase{:d}/Metrics.npz".format(experiment, phase), allow_pickle=True).item()
                training_population = list(np.load("./Delenox_Experiment_Data/{}/Phase{:d}/Training_Set.npz".format(experiment, phase), allow_pickle=True))
            except FileNotFoundError:
                neat_metrics = {'Experiment': experiment, 'Mean Novelty': [], 'Best Novelty': [], 'Node Complexity': [],
                                'Connection Complexity': [],
                                'Archive Size': [], 'Species Count': [], 'Infeasible Size': []}
                training_population = []

        """
        Exploration Phase:
        1) Load the NEAT Generators outputted by the previous iteration (or the given seed if none exist)
        2) Iterate through the generators and run the pre-defined generations of NEAT.
        3) Update the NEAT metrics file and training set of novel individuals, and output to file.
        4) Save the NEAT generator to a pickle file, which in-turn saves the generated lattice through its own function.
        """
        if phase == 0:
            neat_generators = [pickle.load(open("Delenox_Experiment_Data/Seed/Neat_Population_{:d}.pkl".format(runs),
                                                "rb")) for runs in range(runs_per_phase)]
        else:
            neat_generators = [pickle.load(open("Delenox_Experiment_Data/{}/Phase{:d}/Neat_Population_{:d}.pkl".format(experiment, phase-1, runs),
                                                "rb")) for runs in range(runs_per_phase)]

        for number in range(len(neat_generators)):

            if os.path.exists('Delenox_Experiment_Data/{}/Phase{:d}/Neat_Population_{:d}.pkl'.format(experiment, phase, number)):
                continue

            generator, best_fit, metrics = neat_generators[number].run_neat(phase, experiment, static, noise=noisy)
            training_population += list(best_fit)

            # Save the neat populations to pickle files in the current phase folder
            pickle.dump(generator, open("./Delenox_Experiment_Data/{}/Phase{:d}/Neat_Population_{:d}.pkl".format(experiment, phase, number), "wb+"))

            # Save the latest additions to the novel population to a numpy file
            np.save_compressed("./Delenox_Experiment_Data/{}/Phase{:d}/Training_Set.npz".format(experiment, phase), np.asarray(training_population))

            # Update the metrics dictionary with this phase' results
            for key in metrics.keys():
                try:
                    neat_metrics[key].append(metrics[key])
                except KeyError:
                    neat_metrics.update({key: metrics[key]})
                except AttributeError:
                    pass

            # Save the latest metrics to a numpy file for later extraction
            np.save_compressed("./Delenox_Experiment_Data/{}/Phase{:d}/Metrics.npz".format(experiment, phase), neat_metrics)

        """
        Transformation Phase:
        1) If required, rewind back to all previous phases and construct a historical archive of training sets
        2) Create a new auto-encoder with the data generated from this iteration's exploration phase
        3) Visualize metrics of the newly generated auto-encoder and `save` figures to disk
        """
        if not static and not os.path.exists('Delenox_Experiment_Data/{}/Phase{:d}/encoder.json'.format(experiment, phase)):
            training_history = []
            for rewind in range(phase + 1):
                training_history += list(np.load("./Delenox_Experiment_Data/{}/Phase{:d}/Training_Set.npz".format(experiment, rewind)))
            ae, encoder, decoder = create_auto_encoder(model_type=auto_encoder_3d,
                                                       phase=phase,
                                                       experiment=experiment,
                                                       population=np.asarray(training_population),
                                                       noisy=add_noise_parallel(np.asarray(training_population)))

        # Clear the plotting library's cache to make sure we aren't wasting memory
        plt.close('all')
