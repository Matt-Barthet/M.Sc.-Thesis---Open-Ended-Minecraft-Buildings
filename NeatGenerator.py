import os
import pickle
from multiprocessing.pool import Pool
# from sklearn_extra.cluster import KMedoids
from tensorflow.python.keras.utils.np_utils import to_categorical
from Autoencoder import add_noise, convert_to_integer, load_model, create_auto_encoder, auto_encoder_3d
from Delenox_Config import *
from Visualization import *
from Constraints import *
import time
import neat


class NeatGenerator:
    """
    NEAT module of the Delenox pipeline, responsible for creating and evolving CPPN's which are
    used to generate lattices (genotype) that represent buildings (phenotype). Evolution uses
    novelty search as the heuristic, which computes the average Euclidean distance to the k-nearest
    neighbors as well as to an archive of unique novel individuals from past generations.
    """

    def __init__(self, complexity, config, population_id):
        self.current_phase = 0
        self.config = config
        self.config.__setattr__("pop_size", population_size)
        self.config.__getattribute__("genome_config").num_hidden = complexity
        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StatisticsReporter())
        self.encoder = None
        self.decoder = None
        self.archive = {}
        self.current_gen = 0
        self.phase_best_fit = []
        self.population_id = population_id
        self.building_metrics = {"Lattice Stability": {}, "Building Area": {}, "Building Volume": {},
                                 "Bounding Box Volume": {},
                                 "Interior Volume": {}, "Depth Middle": {}, "Width Middle": {}
                                 }

        self.neat_metrics = {'Mean Novelty': [],
                             'Best Novelty': [],
                             'Node Complexity': [],
                             'Connection Complexity': [],
                             'Archive Size': [],
                             'Species Count': [],
                             }
        self.archive_lattices = []
        self.pool = None

    def run_neat(self, phase_number, static=False):
        """
        Executes one "exploration" phase of the Delenox pipeline.  A set number of independent evolutionary runs
        are completed and the top N most novel individuals are taken and inserted into a population.  At the of
        the phase we look at the distribution of individuals in the population according to numerous metrics and
        statistics regarding the evolution of the populations such as the speciation, novelty scores etc.

        :param static:
        :param phase_number:
        :return: the generated population of lattices and statistics variables from the runs of the phase.
        """
        # self.archive.clear()
        self.phase_best_fit.clear()
        # self.archive_lattices.clear()

        self.current_gen = 0
        self.current_phase = phase_number

        if phase_number > 0 and static is False:
            self.encoder = load_model("./Delenox_Experiment_Data/Phase{:d}/encoder".format(phase_number - 1))
            self.decoder = load_model("./Delenox_Experiment_Data/Phase{:d}/decoder".format(phase_number - 1))
        else:
            self.encoder = load_model("./Delenox_Experiment_Data/Seed/encoder")
            self.decoder = load_model("./Delenox_Experiment_Data/Seed/decoder")

        self.pool = Pool(thread_count)
        self.population.run(self.novelty_search_parallel, generations_per_run)
        self.pool.close()
        self.pool.join()

        self.neat_metrics['Mean Novelty'] = self.population.reporters.reporters[0].get_fitness_mean()
        self.neat_metrics['Best Novelty'] = self.population.reporters.reporters[0].get_fitness_stat(max)

        self.pool = None
        self.encoder = None
        self.decoder = None

        return self, self.phase_best_fit, self.neat_metrics

    def novelty_search_parallel(self, genomes, config):
        """
        Multi-process fitness function for the NEAT module of the project.  Implements novelty search and
        scales the workload across the thread count given in the experiment parameters. Assigns a novelty
        value to each genome and keeps the feasible population separate, discarding and randomly regenerating
        the infeasible individuals.

        :param genomes: population of genomes to be evaluated.
        :param config: the NEAT-Python configuration file.
        """
        start = time.time()
        compressed_population = {}
        lattices = {}
        remove = 0

        jobs = []

        for genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(generate_lattice, (genome, config, False, None)))
        for job, (genome_id, genome) in zip(jobs, genomes):
            lattice, _, feasible, metrics = job.get()
            if not feasible:
                del self.population.population[genome_id]
                genome.fitness = 0
                remove += 1
            else:
                lattices.update({genome_id: lattice})

        for genome_id, lattice in lattices.items():
            compressed_population.update({genome_id: self.encoder.predict(lattice[None])[0]})

        jobs.clear()
        for genome_id in compressed_population.keys():
            parameters = (genome_id, compressed_population, self.archive)
            jobs.append(self.pool.apply_async(novelty_search, parameters))
        for job, genome_id in zip(jobs, compressed_population.keys()):
            self.population.population[genome_id].fitness = job.get()

        fitness = {genome_id: fitness.fitness for genome_id, fitness in self.population.population.items() if
                   fitness.fitness > 0}
        sorted_keys = [k for k, _ in sorted(fitness.items(), key=lambda item: item[1])]

        for individual in range(np.min([add_to_archive, len(lattices)])):
            lattice = lattices[sorted_keys[-individual]]
            self.archive_lattices.append(lattice)
            vector = self.encoder.predict(lattice[None])[0]
            if len(self.archive) == 0 or not (vector == list(self.archive.values())).all(1).any():
                self.archive.update({sorted_keys[-individual]: vector})

        if self.current_gen % 100 == 0 or self.current_gen + 1 == generations_per_run:
            most_novel_lattice = lattices[sorted_keys[-1]]
            least = lattices[sorted_keys[0]]
            mid = lattices[sorted_keys[int(len(sorted_keys) / 2)]]
            novelty_voxel_plot(
                [convert_to_integer(least), convert_to_integer(mid), convert_to_integer(most_novel_lattice)],
                self.current_gen + 1, self.population_id, self.current_phase)

        if self.current_gen + 1 == generations_per_run:
            np.save("./Delenox_Experiment_Data/Phase{:d}/Population_{:d}.npy".format(self.current_phase,
                                                                                     self.population_id), lattices)
            for individual in range(np.min([best_fit_count, len(lattices)])):
                self.phase_best_fit.append(lattices[sorted_keys[-individual]])

        node_complexity = 0
        connection_complexity = 0

        for individual in self.population.population.values():
            node_complexity += individual.size()[0]
            connection_complexity += individual.size()[1]

        node_complexity /= len(self.population.population)
        connection_complexity /= len(self.population.population)

        self.neat_metrics['Node Complexity'].append(node_complexity)
        self.neat_metrics['Connection Complexity'].append(connection_complexity)
        self.neat_metrics['Archive Size'].append(len(self.archive))
        self.neat_metrics['Species Count'].append(len(self.population.species.species))

        print("[Population {:d}]: Generation {:d} took {:2f} seconds.".format(self.population_id, self.current_gen,
                                                                              time.time() - start))
        print("Average Hidden Layer Size: {:2.2f}".format(node_complexity))
        print("Average Connection Count: {:2.2f}".format(connection_complexity))
        print("Size of the Novelty Archive: {:d}".format(len(self.archive)))
        print("Number of Infeasible Buildings:", remove, "\n")

        self.current_gen += 1


def voxel_based_diversity(genome_id, lattice, lattices):
    pixel_diversity = 0
    for compare in lattices:
        for (x, y, z) in value_range:
            if lattice[x][y][z].all() != compare[x][y][z].all():
                pixel_diversity += 1
    return {genome_id: pixel_diversity}


def novelty_search(genome_id, compressed_population, archive):
    """
    Computes the novelty score for the given genome with respect to the current population and
    an archive of past novel individuals for this run. The score is the average euclidean distance
    to the nearest K neighbors (taken from the population and archive).

    :param genome_id: the ID of the genome being assessed.
    :param compressed_population: the population of latent vectors to compare to.
    :param archive: the archive of past novel individuals for this run.
    :return: the novelty score for this genome.
    """
    distances = []
    for neighbour in list(compressed_population.values()) + list(archive.values()):
        distance = 0
        for element in range(len(neighbour)):
            distance += np.square(compressed_population[genome_id][element] - neighbour[element])
        distances.append(np.sqrt(distance))
    return np.round(np.average(distances[:k_nearest_neighbors]), 2)


def generate_lattice(genome, config, noise_flag=True, plot=None):
    """

    :param plot:
    :param noise_flag:
    :param genome:
    :param config:
    :return:
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    lattice = np.zeros(lattice_dimensions)
    noisy = np.zeros(lattice_dimensions)

    for (x, y, z) in value_range:
        lattice[x][y][z] = np.round(
            net.activate((x / lattice_dimensions[0], y / lattice_dimensions[0], z / lattice_dimensions[0]))[0])
    feasible, lattice, metrics = apply_constraints(lattice)
    if noise_flag:
        noisy = add_noise(lattice)
    if plot is not None:
        voxel_plot(lattice, plot)

    lattice = to_categorical(lattice, num_classes=5)
    noisy = to_categorical(noisy, num_classes=5)

    return np.asarray(lattice, dtype=bool), np.asarray(noisy, dtype=bool), feasible, metrics


def generate_lattices(genomes, config, noise_flag=True):
    """

    :param noise_flag:
    :param genomes:
    :param config:
    :return:
    """
    pool = Pool(thread_count)
    jobs = []
    lattices = []
    noisy = []
    metrics = []
    for genome in genomes:
        jobs.append(pool.apply_async(generate_lattice, (genome, config, noise_flag)))
    for job in jobs:
        lattice, noisy_lattice, valid, metrics = job.get()
        if valid:
            lattices.append(lattice)
            if noise_flag:
                noisy.append(noisy_lattice)
    pool.close()
    pool.join()
    return noisy, lattices, metrics


def create_population_lattices(config, noise_flag=True):
    """
    Generates a population of lattices and their noisy counterparts.

    :param noise_flag: boolean which determines whether a noised copy of the dataset should be created.
    :param config: CPPN-NEAT config file specifying the parameters for the genomes.
    :return lattices, noisy: the population of generated lattices and their noisy counterparts
    """
    lattices = []
    noisy = []
    while len(lattices) < best_fit_count * runs_per_phase:
        population = create_population(config, round((best_fit_count * runs_per_phase - len(lattices)) * 2))
        noisy_batch, lattice_batch, _ = generate_lattices(population.population.values(), config, noise_flag)
        lattices += lattice_batch
        if noise_flag:
            noisy += noisy_batch
        print("{:d} Lattices created!".format(len(lattices)))
    return np.asarray(lattices[:1000], dtype=bool), np.asarray(noisy[:1000], dtype=bool)


def create_population(config, pop_size=population_size):
    """
    Generates a population of CPPN genomes according to the given CPPN-NEAT config file and population size.

    :param config: CPPN-NEAT config file specifying the parameters for the genomes.
    :param pop_size: Number of genomes to create.
    :return population: Population objecting containing a dictionary in the form {genome_id: genome_object}.
    """
    config.__setattr__("pop_size", pop_size)
    population = neat.Population(config)
    return population


def create_seed_files(config):
    """

    :param config:
    :return:
    """
    training_population, _ = create_population_lattices(config, False)
    np.save("./Delenox_Experiment_Data/Seed/Initial_Training_Set.npy", np.asarray(training_population))
    _ = create_auto_encoder(model_type=auto_encoder_3d,
                            phase=0,
                            population=np.asarray(training_population),
                            noisy=None)
    for runs in range(runs_per_phase):
        generator = NeatGenerator(
            config=config,
            complexity=0,
            population_id=runs
        )
        with open("Neat_Population_{:d}".format(runs), "wb+") as f:
            pickle.dump(generator, f)


def cluster_analysis(population, metrics, title, axis_labels, config):
    """

    :param population:
    :param metrics:
    :param title:
    :param axis_labels:
    :param config:
    :return:
    """
    clustering = KMedoids(n_clusters=5)
    data = np.asarray(list(zip(list(metrics[0].values()), list(metrics[1].values()))))
    data_dict = {k: [d[k] for d in metrics] for k in metrics[0].keys()}
    clustering.fit(data)
    clusters = clustering.predict(data)
    medoids = clustering.cluster_centers_

    for medoid in medoids:
        for genome, metrics in data_dict.items():
            if list(medoid) == list(metrics):
                medoid_lattice = generate_lattice(population[genome], config, False)[0][0]
                voxel_plot(convert_to_integer(medoid_lattice), "Medoid at " + str(list(medoid)), "")
                break

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=clusters, s=50, cmap='viridis')
    plt.scatter(medoids[:, 0], medoids[:, 1], c='black', s=200, alpha=0.5)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.savefig("./Delenox_Experiment_Data/Run" + str(current_run) + "/Clustering_" + str(time.time()) + ".png")
    plt.show()
