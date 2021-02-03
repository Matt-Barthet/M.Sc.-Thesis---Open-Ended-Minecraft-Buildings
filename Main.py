import os
import neat
import tensorflow as tf
import matplotlib.pyplot as plt
from Autoencoder import auto_encoder_3d, create_auto_encoder
from Delenox_Config import *
from NeatGenerator import NeatGenerator, create_population_lattices
from Visualization import plot_statistics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load configuration file according to the given path and setting relevant parameters.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, config_path)
    config.genome_config.add_activation('sin_adjusted', sinc)

    encoders = []
    decoders = []
    populations = []

    training_population, _ = create_population_lattices(config, False)
    training_history = []

    # Initialise a set of neat populations that will be evolved in the Exploration Phases.
    neat_generators = []
    processes = []

    for runs in range(runs_per_phase):
        neat_generators.append(NeatGenerator(
            config=config,
            complexity=0,
            population_id=runs
        ))

    for phase in range(number_of_phases):
        plt.close('all')

        training_history += list(training_population)
        training_population = np.asarray(training_population)
        ae, encoder, decoder = create_auto_encoder(model_type=auto_encoder_3d,
                                                   phase=phase,
                                                   population=np.asarray(training_history),
                                                   noisy=None)

        np.save("./Delenox_Experiment_Data/Phase{:d}/Training_Set.npy".format(phase),
                np.asarray(training_population))

        training_population = list(training_population)
        training_population.clear()

        # Execute the exploration phase and get the resulting population of novel individuals and statistics.
        neat_metrics = {'Mean Novelty': [],
                        'Best Novelty': [],
                        'Node Complexity': [],
                        'Connection Complexity': [],
                        'Archive Size': [],
                        'Species Count': [],
                        }

        processes = []
        queues = []

        for number in range(len(neat_generators)):
            (generator, best_fit, metrics) = neat_generators[number].run_neat(phase)
            neat_generators[generator.population_id] = generator
            training_population += list(best_fit)
            for key in metrics.keys():
                neat_metrics[key].append(metrics[key])

        # Visualize the data retrieved for the exploration phase.
        for key in neat_metrics.keys():
            neat_metrics[key] = np.stack((neat_metrics[key]), axis=-1)
            plot_statistics(
                values=np.mean(neat_metrics[key], axis=-1),
                confidence=np.std(neat_metrics[key], axis=-1),
                key=key,
                phase=phase
            )

        np.save("./Delenox_Experiment_Data/Phase{:d}/Metrics.npy".format(phase), neat_metrics)
        current_run += 1
