import os
import neat
from sklearn.decomposition import PCA

from NeatGenerator import NeatGenerator, create_population, create_population_lattices
from Autoencoder import auto_encoder_3d, load_model, create_auto_encoder, update_auto_encoder, add_noise_parallel, \
    test_accuracy
from Visualization import plot_statistics
from Delenox_Config import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import pandas as pd
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

    population_history = []
    autoencoders = []

    initial_population, noisy = create_population_lattices(config)
    ae, encoder, decoder = create_auto_encoder(compressed_length, auto_encoder_3d, (initial_population, noisy))
    autoencoders.append({"model": ae, "encoder": encoder, "decoder": decoder})
    current_complexity = 0

    for phase in range(number_of_phases):

        # Initialise a neat generator for the exploration phase
        neat_generator = NeatGenerator(
            encoder=encoder,
            decoder=decoder,
            config=config,
            generations=generations_per_run,
            k=k_nearest_neighbors,
            complexity=current_complexity,
            latent_size=compressed_length
        )

        # Execute the exploration phase and get the resulting population of novel individuals and statistics.
        best_fit, new_population, neat_metrics = neat_generator.run_neat()
        population_history += best_fit

        for _, individual in new_population.items():
            current_complexity = int(np.mean([current_complexity, individual.size()[0]]))

        # Visualize the data retrieved for the exploration phase.
        plot_statistics(
            metrics=neat_metrics,
            keys=['Best Novelty', 'Mean Novelty', 'Node Complexity', 'Connection Complexity', 'Archive Size'],
            current_run=current_run
        )

        # Transformation phase: create and train a new autoencoder based on the previous exploration phase
        ae, encoder, decoder = create_auto_encoder(compressed_length, auto_encoder_3d, (np.asarray(best_fit), add_noise_parallel(best_fit)))
        autoencoders.append({"model": ae, "encoder": encoder, "decoder": decoder})

        # ae, encoder, decoder = create_auto_encoder(256, auto_encoder_3d, population_history)
        # ae, = update_auto_encoder(ae, list(population.values()))

        plt.figure()
        plt.title("PCA Analysis of Compressed Buildings")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        compressed = [[] for _ in range(len(autoencoders))]
        for model in range(len(autoencoders)):
            for lattice in best_fit:
                compressed[model].append(autoencoders[model]['encoder'].predict(lattice[None])[0])
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(compressed[model])
            principalDf = pd.DataFrame(data=principalComponents, columns=['1', '2'])
            plt.scatter(principalDf['1'], principalDf['2'], cmap=[model] * len(principalDf))
        plt.savefig("./Delenox_Experiment_Data/Run{}/PCA.png".format(current_run))

        eucl_averages = []
        for model in range(len(autoencoders)):
            average = 0
            for vector in compressed[model]:
                for other in compressed[model]:
                    dist = np.linalg.norm(vector - other)
                    average = np.mean([average, dist])
            eucl_averages.append(average)
        plt.figure()
        plt.bar(x=range(1, len(eucl_averages) + 1), height=eucl_averages)
        plt.xlabel("Autoencoder from Phase")
        plt.ylabel("Euclidean Distance")
        plt.title("Average Euclidean Distance (Vectors) on latest Novel Batch")
        plt.savefig("./Delenox_Experiment_Data/Run{}/Eucl_Latest.png".format(current_run))

        inital_compressed = [[] for _ in range(len(autoencoders))]
        eucl_averages = []
        for model in range(len(autoencoders)):
            for lattice in initial_population:
                inital_compressed[model].append(autoencoders[model]['encoder'].predict(lattice[None])[0])
            average = 0
            for vector in inital_compressed[model]:
                for other in compressed[model]:
                    dist = np.linalg.norm(vector - other)
                    average = np.mean([average, dist])
            eucl_averages.append(average)
        plt.figure()
        plt.bar(x=range(1, len(eucl_averages) + 1), height=eucl_averages)
        plt.xlabel("Autoencoder from Phase")
        plt.ylabel("Euclidean Distance")
        plt.title("Average Euclidean Distance (Vectors) on Initial Population")
        plt.savefig("./Delenox_Experiment_Data/Run{}/Eucl_Initial.png".format(current_run))

        errors = []
        for model in range(len(autoencoders)):
            error = test_accuracy(autoencoders[model]["encoder"], autoencoders[model]["decoder"], rn.sample(best_fit, 250))
            errors.append(error)
        plt.figure()
        plt.bar(x=range(1, len(errors) + 1), height=errors)
        plt.xlabel("Autoencoder from Phase")
        plt.ylabel("Error %")
        plt.title("Autoencoder Reconstruction Error on Latest Novel Batch")
        plt.savefig("./Delenox_Experiment_Data/Run{}/Error_Latest.png".format(current_run))

        errors = []
        for model in range(len(autoencoders)):
            error = test_accuracy(autoencoders[model]["encoder"], autoencoders[model]["decoder"], rn.sample(list(initial_population), 250))
            errors.append(error)
        plt.figure()
        plt.bar(x=range(1, len(errors) + 1), height=errors)
        plt.xlabel("Autoencoder from Phase")
        plt.ylabel("Error %")
        plt.title("Autoencoder Reconstruction Error on Initial Population")
        plt.savefig("./Delenox_Experiment_Data/Run{}/Error_Initial.png".format(current_run))

        current_run += 1

    # np.save("./Novelty_Experiments/Neat_Experiment_No_Constraints.npy", np.asarray([neat_means, neat_means_std, neat_bests, neat_bests_std]))
