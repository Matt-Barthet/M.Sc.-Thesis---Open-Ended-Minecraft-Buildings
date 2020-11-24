from multiprocessing.pool import Pool
from sklearn_extra.cluster import KMedoids
from tensorflow.python.keras.utils.np_utils import to_categorical
from Autoencoder import add_noise, convert_to_integer
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
    def __init__(self, encoder, decoder, config, generations, k, latent_size, complexity):
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.generations = generations
        self.neighbors = k
        self.means_list = [[] for _ in range(generations_per_run)]
        self.std_list = [[] for _ in range(generations_per_run)]
        self.bests_list = [[] for _ in range(generations_per_run)]
        self.compressed_length = latent_size
        self.archive = {}
        self.current_gen = 0
        self.complexity = complexity
        self.phase_best_fit = []
        self.phase_best_genomes = {}
        self.population = None
        self.compressed_population = {}

        self.metrics = {"Lattice Stability": {}, "Building Area": {}, "Building Volume": {}, "Bounding Box Volume": {},
                        "Interior Volume": {}, "Depth Middle": {}, "Width Middle": {}}

    def run_neat(self):
        """
        Executes one "exploration" phase of the Delenox pipeline.  A set number of independent evolutionary runs
        are completed and the top N most novel individuals are taken and inserted into a population.  At the of
        the phase we look at the distribution of individuals in the population according to numerous metrics and
        statistics regarding the evolution of the populations such as the speciation, novelty scores etc.

        :return: the generated population of lattices and statistics variables from the runs of the phase.
        """
        for run in range(runs_per_phase):
            print("\nStarting Run: ", run + 1)
            self.current_gen = 0
            self.archive.clear()
            self.population = neat.Population(self.config)
            self.population.add_reporter(neat.StdOutReporter(True))
            self.population.add_reporter(neat.StatisticsReporter())
            self.config.__setattr__("pop_size", population_size)
            self.config.__setattr__("num_hidden", self.complexity)
            self.population.run(self.novelty_search_parallel, self.generations)
            means = self.population.reporters.reporters[1].get_fitness_mean()
            bests = self.population.reporters.reporters[1].get_fitness_stat(max)

            for generation in range(generations_per_run):
                self.means_list[generation].append(means[generation])
                self.bests_list[generation].append(bests[generation])

        means = []
        means_confidence = []
        bests = []
        bests_confidence = []

        for generation in range(generations_per_run):
            means.append(np.mean(self.means_list[generation]))
            means_confidence.append(np.std(self.means_list[generation]))
            bests.append(np.mean(self.bests_list[generation]))
            bests_confidence.append(np.std(self.bests_list[generation]))

        return self.phase_best_fit, means, means_confidence, bests, bests_confidence, self.phase_best_genomes

    def novelty_search_parallel(self, genomes, config):
        """
        Multi-process fitness function for the NEAT module of the project.  Implements novelty search and
        scales the workload across the thread count given in the experiment parameters. Assigns a novelty
        value to each genome and keeps the feasible population separate, discarding and randomly regenerating
        the infeasible individuals.

        :param genomes: population of genomes to be evaluated.
        :param config: the NEAT-Python configuration file.
        """
        pool = Pool(thread_count)
        jobs = []

        self.compressed_population.clear()
        lattices = {}
        remove = []

        metrics_this_run = {"Lattice Stability": {}, "Building Area": {}, "Building Volume": {},
                            "Bounding Box Volume": {}, "Interior Volume": {}, "Depth Middle": {}, "Width Middle": {}}

        start = time.time()
        print("(CPU x " + str(thread_count) + "): Generating lattices and applying filters...", end="")
        for genome_id, genome in genomes:
            jobs.append(pool.apply_async(generate_lattice, (genome_id, genome, config, False, None)))
        for job, (genome_id, genome) in zip(jobs, genomes):
            key_pair, _, feasible, metrics = job.get()
            if not feasible:
                remove.append(genome_id)
                genome.fitness = 0
            else:
                lattices.update(key_pair)
                for key in metrics_this_run.keys():
                    metrics_this_run[key].update({genome_id: metrics[key]})

        print("Done! (", np.round(time.time() - start, 2), "seconds ).")

        start = time.time()
        print("(GPU): Compressing the generated lattices using encoder model in main thread...", end="")
        for lattice_id, lattice in lattices.items():
            compressed = self.encoder.predict(lattice[None])[0]
            self.compressed_population.update({lattice_id: compressed})
        print("Done! (", np.round(time.time() - start, 2), "seconds ).")

        jobs.clear()
        start = time.time()
        print("(CPU x " + str(thread_count) + "): Starting novelty search on compressed population...", end="")
        for genome_id in self.compressed_population.keys():
            parameters = (genome_id, self.compressed_population, self.neighbors, self.compressed_length, self.archive)
            jobs.append(pool.apply_async(novelty_search, parameters))
        for job, genome_id in zip(jobs, self.compressed_population.keys()):
            self.population.population[genome_id].fitness = job.get()

        print("Done! (", np.round(time.time() - start, 2), "seconds ).")

        fitness = {genome_id: fitness.fitness for genome_id, fitness in self.population.population.items() if
                   fitness.fitness > 0}
        sorted_keys = [k for k, _ in sorted(fitness.items(), key=lambda item: item[1])]

        for individual in range(add_to_archive):
            lattice = lattices[sorted_keys[-individual]]
            vector = self.encoder.predict(lattice[None])[0]
            if len(self.archive) == 0 or not (vector == list(self.archive.values())).all(1).any():
                self.archive.update({sorted_keys[-individual]: vector})

        """if self.current_gen % 10 == 0 or self.current_gen +1 == self.generations:
            most_novel_lattice = lattices[sorted_keys[-1]]
            least = lattices[sorted_keys[0]]
            mid = lattices[sorted_keys[int(len(sorted_keys) / 2)]]
            novelty_voxel_plot([convert_to_integer(least), convert_to_integer(mid), convert_to_integer(most_novel_lattice)],
                               self.current_gen + 1)"""

        """jobs.clear()
        for genome_id, lattice in lattices.items():
            jobs.append(pool.apply_async(voxel_based_diversity, (genome_id, lattice, lattices.values())))
        for job in jobs:
            lattices_diversities.update(job.get())"""

        pool.close()
        pool.join()

        print("\nNumber of Infeasible Buildings:", len(remove), "\n")

        if self.current_gen + 1 == generations_per_run:

            for individual in range(best_fit_count):
                lattice = lattices[sorted_keys[-individual]]
                self.phase_best_fit.append(lattice)
                self.phase_best_genomes.update({len(self.phase_best_genomes): self.population.population[sorted_keys[-individual]]})

        for key in remove:
            del self.population.population[key]
        self.current_gen += 1


def voxel_based_diversity(genome_id, lattice, lattices):
    pixel_diversity = 0
    for compare in lattices:
        for (x, y, z) in value_range:
            if lattice[x][y][z].all() != compare[x][y][z].all():
                pixel_diversity += 1
    return {genome_id: pixel_diversity}


def novelty_search(genome_id, compressed_population, k, compressed_length, archive):
    """
    Computes the novelty score for the given genome with respect to the current population and
    an archive of past novel individuals for this run. The score is the average euclidean distance
    to the nearest K neighbors (taken from the population and archive).

    :param genome_id: the ID of the genome being assessed.
    :param compressed_population: the population of latent vectors to compare to.
    :param k: the number of nearest neighbors to average over when calculating the final score.
    :param compressed_length: the length of the compressed representation.
    :param archive: the archive of past novel individuals for this run.
    :return: the novelty score for this genome.
    """
    distances = []
    for neighbour in list(compressed_population.values()) + list(archive.values()):
        distance = 0
        for element in range(compressed_length):
            distance += np.square(compressed_population[genome_id][element] - neighbour[element])
        distances.append(np.sqrt(distance))
    return np.round(np.average(distances[:k]), 2)


def generate_lattice(genome_id, genome, config, noise_flag=True, plot=None):
    """

    :param genome_id:
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
        lattice[x][y][z] = np.round(net.activate((x / lattice_dimensions[0], y / lattice_dimensions[0], z / lattice_dimensions[0]))[0])
    feasible, lattice, metrics = apply_constraints(lattice)
    if noise_flag:
        noisy = add_noise(lattice)
    if plot is not None:
        voxel_plot(lattice, plot)

    lattice = to_categorical(lattice, num_classes=5)
    noisy = to_categorical(noisy, num_classes=5)

    return {genome_id: np.asarray(lattice)}, np.asarray(noisy), feasible, metrics


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
        jobs.append(pool.apply_async(generate_lattice, (0, genome, config, noise_flag)))
    for job in jobs:
        lattice, noisy_lattice, valid, metrics = job.get()
        if valid:
            lattices.append(list(lattice.values())[0])
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
        population = create_population(config, round((best_fit_count * runs_per_phase - len(lattices)) * 1.75))
        noisy_batch, lattice_batch, _ = generate_lattices(population.population.values(), config, noise_flag)
        lattices += lattice_batch
        if noise_flag:
            noisy += noisy_batch
    return np.asarray(lattices[:best_fit_count * runs_per_phase]), np.asarray(noisy[:best_fit_count * runs_per_phase])


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
                medoid_lattice = generate_lattice(0, population[genome], config, False)[0][0]
                voxel_plot(convert_to_integer(medoid_lattice), "Medoid at " + str(list(medoid)), "")
                break

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=clusters, s=50, cmap='viridis')
    plt.scatter(medoids[:, 0], medoids[:, 1], c='black', s=200, alpha=0.5)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    plt.savefig("./Delenox_Experiment_Data/Run"+str(current_run)+"/Clustering_"+str(time.time())+".png")
    plt.show()
