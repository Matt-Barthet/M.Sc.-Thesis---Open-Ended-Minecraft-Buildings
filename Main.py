import os
import sys
from multiprocessing import Pool
from multiprocessing.context import Process

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

    # Initialise a set of neat populations that will be evolved in the Exploration Phases.
    neat_generators = []
    for runs in range(runs_per_phase):
        neat_generators.append(NeatGenerator(
            config=config,
            complexity=0,
            population_id=runs
        ))

    training_population, _ = create_population_lattices(config, False)

    for phase in range(0, number_of_phases):
        """
        Transformation Phase:
            Retrieve the most novel X individuals from each population from the previous exploration phase, 
            and train an autoencoder on them. The training population and trained model are saved to disk 
            as part of the experiment results.
        Note: A noisy copy of the training population are used as input for training.
        """
        ae, encoder, decoder = create_auto_encoder(model_type=auto_encoder_3d,
                                                   phase=phase,
                                                   population=training_population,
                                                   noisy=add_noise_parallel(training_population))

        np.save("./Delenox_Experiment_Data/Phase{:d}/Training_Set.npy".format(phase),
                np.asarray(training_population))

        print(sys.getsizeof(ae))
        print(sys.getsizeof(encoder))
        print(sys.getsizeof(decoder))

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

        process = Process(target=neat_generators[0].run_neat, args=(phase,))
        process.start()
        process.join()

        for key in neat_metrics.keys():
            neat_metrics[key] = np.stack((neat_metrics[key]), axis=-1)
            # Visualize the data retrieved for the exploration phase.
            plot_statistics(
                values=np.mean(neat_metrics[key], axis=-1),
                confidence=np.std(neat_metrics[key], axis=-1),
                key=key,
                phase=phase
            )

        plt.figure()
        plt.title("PCA Analysis of Compressed Buildings")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        compressed = [[] for _ in range(len(autoencoders))]
        for model in range(len(autoencoders)):
            for lattice in next_population:
                compressed[model].append(autoencoders[model]['encoder'].predict(lattice[None])[0])
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(compressed[model])
            principalDf = pd.DataFrame(data=principalComponents, columns=['1', '2'])
            plt.scatter(principalDf['1'], principalDf['2'], cmap=[model] * len(principalDf), s=10, label="AE Phase {:d}".format(phase))
            plt.legend()
        plt.savefig("./Delenox_Experiment_Data/Phase{}/PCA.png".format(current_run))

        errors = []
        for model in range(len(autoencoders)):
            error = test_accuracy(autoencoders[model]["encoder"], autoencoders[model]["decoder"],
                                  rn.sample(list(next_population), 250))
            errors.append(error)
        plt.figure()
        plt.bar(x=range(1, len(errors) + 1), height=errors)
        plt.xlabel("Autoencoder from Phase")
        plt.ylabel("Error %")
        plt.title("Autoencoder Reconstruction Error on Latest Novel Batch")
        plt.savefig("./Delenox_Experiment_Data/Phase{}/Error_Latest.png".format(current_run))

        """eucl_averages = []
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
        plt.savefig("./Delenox_Experiment_Data/Run{}/Eucl_Latest.png".format(current_run))"""

        """inital_compressed = [[] for _ in range(len(autoencoders))]
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
        plt.savefig("./Delenox_Experiment_Data/Run{}/Eucl_Initial.png".format(current_run))"""

        """errors = []
        for model in range(len(autoencoders)):
            error = test_accuracy(autoencoders[model]["encoder"], autoencoders[model]["decoder"],
                                  rn.sample(list(initial_population), 250))
            errors.append(error)
        plt.figure()
        plt.bar(x=range(1, len(errors) + 1), height=errors)
        plt.xlabel("Autoencoder from Phase")
        plt.ylabel("Error %")
        plt.title("Autoencoder Reconstruction Error on Initial Population")
        plt.savefig("./Delenox_Experiment_Data/Run{}/Error_Initial.png".format(current_run))"""

        current_run += 1

    # np.save("./Novelty_Experiments/Neat_Experiment_No_Constraints.npy", np.asarray([neat_means, neat_means_std, neat_bests, neat_bests_std]))
