import os
import neat
from NeatGenerator import NeatGenerator, create_population, create_population_lattices
from Autoencoder import auto_encoder_3d, load_model, create_auto_encoder, update_auto_encoder, add_noise_parallel, \
    test_accuracy
from Visualization import plot_statistics
from Delenox_Config import *
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def sinc(x):
    x = max(-60.0, min(60.0, 5.0 * x))
    return (np.sin(x) + 1) / 2


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load configuration file according to the given path and setting relevant parameters.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    config.__setattr__("pop_size", population_size)
    config.genome_config.add_activation('sin_adjusted', sinc)

    encoder = load_model("material_encoder_256")
    decoder = load_model("material_decoder_256")
    population_history = []

    # initial_population, noisy = create_population_lattices(config)
    # ae, encoder, decoder = create_auto_encoder(256, auto_encoder_3d, (initial_population, noisy))

    current_complexity = 0

    for phase in range(number_of_phases):

        # Initialise a neat generator for the exploration phase
        neat_generator = NeatGenerator(
            encoder=encoder,
            decoder=decoder,
            config=config,
            generations=generations_per_run,
            k=k_nearest_neighbors,
            complexity = current_complexity,
            latent_size=compressed_length
        )

        # Execute the exploration phase and get the resulting population of novel individuals and statistics.
        best_fit, neat_means, neat_means_std, neat_bests, neat_bests_std, new_population = neat_generator.run_neat()
        population_history += best_fit

        for id, individual in new_population.items():
            current_complexity = np.max([current_complexity, individual.size()[0]])
        print(current_complexity)

        # Visualize the data retrieved for the exploration phase.
        plot_statistics(
            generations=generations_per_run,
            bests=[neat_bests],
            bests_confidence=[neat_bests_std],
            means=[neat_means],
            means_confidence=[neat_means_std],
            names=["Neat"],
            averaged_runs=runs_per_phase
        )

        # Transformation phase: create and train a new autoencoder based on the previous exploration phase
        ae, encoder, decoder = create_auto_encoder(256, auto_encoder_3d, (np.asarray(best_fit), add_noise_parallel(best_fit)))

        # ae, encoder, decoder = create_auto_encoder(256, auto_encoder_3d, population_history)
        # ae, = update_auto_encoder(ae, list(population.values()))
        test_accuracy(encoder, decoder, population_history)
        current_run += 1

    # np.save("./Novelty_Experiments/Neat_Experiment_No_Constraints.npy", np.asarray([neat_means, neat_means_std, neat_bests, neat_bests_std]))
