from multiprocessing.pool import Pool
from tensorflow.python.keras.utils.np_utils import to_categorical
from Autoencoder import add_noise, test_accuracy, convert_to_integer
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
    def __init__(self, encoder, decoder, config, generations, k, num_workers, compressed_length):
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.generations = generations
        self.population = None
        self.compressed_population = {}
        self.neighbors = k
        self.num_workers = num_workers
        self.means_list = [[] for _ in range(latent_generations)]
        self.std_list = [[] for _ in range(latent_generations)]
        self.bests_list = [[] for _ in range(latent_generations)]
        self.compressed_length = compressed_length
        self.archive = {}
        self.current_gen = 0
        self.interior_space_ratios = []
        self.floor_to_ceiling_ratios = []
        self.building_to_lattice_ratios = []

    def run_neat(self):
        """
        TODO: Implement the Feasible-Infeasible split and evolve the infeasible population by minimizing the
              euclidean distance to the feasible threshold.
        :return:
        """

        for run in range(averaged_runs):
            
            print("\nStarting Run: ", run + 1)
            self.current_gen = 0
            self.config.__setattr__("pop_size", best_fit_count)
            self.population = neat.Population(self.config)
            self.population.add_reporter(neat.StdOutReporter(True))
            self.population.add_reporter(neat.StatisticsReporter())
            self.population.run(self.novelty_search_parallel, self.generations)
            means = self.population.reporters.reporters[1].get_fitness_mean()
            bests = self.population.reporters.reporters[1].get_fitness_stat(max)

            for generation in range(latent_generations):
                self.means_list[generation].append(means[generation])
                self.bests_list[generation].append(bests[generation])

            histogram, x_edges, y_edges = np.histogram2d(x=self.floor_to_ceiling_ratios, y=self.building_to_lattice_ratios, bins=np.linspace(0, 1, 10))

            fig = plt.figure()
            plt.title("Expressive Range of Generator")
            plt.xlabel("Floor Voxels : Ceiling Voxels")
            plt.ylabel("Total Voxels : Lattice Volume")
            pops = plt.imshow(histogram, interpolation='nearest', origin='low', aspect='auto',
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap=plt.cm.get_cmap("gray"))
            fig.colorbar(pops, label="Building Frequency")
            plt.show()

        means = []
        means_confidence = []
        bests = []
        bests_confidence = []

        for generation in range(latent_generations):
            means.append(np.mean(self.means_list[generation]))
            means_confidence.append(np.std(self.means_list[generation]))
            bests.append(np.mean(self.bests_list[generation]))
            bests_confidence.append(np.std(self.bests_list[generation]))

        return self.population.population, means, means_confidence, bests, bests_confidence

    def novelty_search_parallel(self, genomes, config):
        """
        Multi-process fitness function for the NEAT module of the project.  Implements novelty search and
        scales the workload across the thread count given in the experiment parameters. Assigns a novelty
        value to each genome and keeps the feasible population separate, discarding and randomly regenerating
        the infeasible individuals.

        :param genomes: population of genomes to be evaluated.
        :param config: the NEAT-Python configuration file.
        """
        lattices = {}
        remove = []
        pool = Pool(thread_count)
        jobs = []
        self.compressed_population.clear()
        self.interior_space_ratios.clear()
        self.floor_to_ceiling_ratios.clear()
        self.building_to_lattice_ratios.clear()

        start = time.time()
        print("(CPU x " + str(thread_count) + "): Generating lattices and applying filters...", end=""),
        for genome_id, genome in genomes:
            jobs.append(pool.apply_async(generate_lattice, (genome_id, genome, config, False, None)))
        for job, (genome_id, genome) in zip(jobs, genomes):
            key_pair, _, feasible, metrics = job.get()
            if not feasible:
                remove.append(genome_id)
                genome.fitness = 0
            else:
                lattices.update(key_pair)
                self.interior_space_ratios.append(metrics[0])
                self.floor_to_ceiling_ratios.append(metrics[1])
                self.building_to_lattice_ratios.append(metrics[2])

        print("Done! (", np.round(time.time() - start, 2), "seconds ).")

        start = time.time()
        print("(GPU): Compressing the generated lattices using encoder model in main thread...", end=""),
        for lattice_id, lattice in lattices.items():
            compressed = self.encoder.predict(lattice[None])[0]
            self.compressed_population.update({lattice_id: compressed})
            # reconstructed = np.round(self.decoder.predict(compressed[None])[0])
            # auto_encoder_plot(lattice, compressed, reconstructed)
        print("Done! (", np.round(time.time() - start, 2), "seconds ).")

        start = time.time()
        print("(CPU x " + str(thread_count) + "): Starting novelty search on compressed population...", end=""),
        jobs.clear()
        for genome_id in self.compressed_population.keys():
            parameters = (genome_id, self.compressed_population, self.neighbors, self.compressed_length, self.archive)
            jobs.append(pool.apply_async(novelty_search, parameters))
        for job, genome_id in zip(jobs, self.compressed_population.keys()):
            self.population.population[genome_id].fitness = job.get()
        print("Done! (", np.round(time.time() - start, 2), "seconds ).")

        pool.close()
        pool.join()

        print("Number of Invalid Lattices: " + str(len(remove)))

        fitness = {genome_id: fitness.fitness for genome_id, fitness in self.population.population.items() if
                   fitness.fitness > 0}
        sorted_keys = [k for k, _ in sorted(fitness.items(), key=lambda item: item[1])]
        most_novel_genome = self.population.population[sorted_keys[-1]]
        most_novel_lattice = generate_lattice(0, most_novel_genome, self.config, noise_flag=False)[0]
        most_novel_vector = self.encoder.predict(most_novel_lattice[0][None])[0]
        self.archive.update({sorted_keys[-1]: most_novel_vector})

        """if self.current_gen % 5 == 0:
            least = generate_lattice(0, self.population.population[sorted_keys[0]], self.config, noise_flag=False)[0]
            mid = generate_lattice(0, self.population.population[sorted_keys[int(len(sorted_keys) / 2)]], self.config, noise_flag=False)[0]
            novelty_voxel_plot([convert_to_integer(least[0]), convert_to_integer(mid[0]), convert_to_integer(most_novel_lattice[0])], self.current_gen + 1)
            test_accuracy(self.encoder, self.decoder, [least[0], mid[0], most_novel_lattice[0]])"""

        for key in remove:
            del self.population.population[key]
        self.current_gen += 1


def novelty_search(genome_id, compressed_population, k, compressed_length, archive):
    """

    :param genome_id:
    :param compressed_population:
    :param k:
    :param compressed_length:
    :param archive:
    :return:
    """
    distances = []
    for neighbour in list(compressed_population.values()):
        distance = 0
        for element in range(compressed_length):
            distance += np.square(compressed_population[genome_id][element] - neighbour[element])
        distances.append(np.sqrt(distance))
    distances.sort()
    k_average = np.round(np.average(distances[:k]), 2)

    if len(archive) == 0:
        return k_average

    distances.clear()
    for neighbour in list(archive.values()):
        distance = 0
        for element in range(compressed_length):
            distance += np.square(compressed_population[genome_id][element] - neighbour[element])
        distances.append(np.sqrt(distance))
    archive_average = np.round(np.average(distances), 2)

    return np.round(np.average([k_average, archive_average]), 2)


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
    while len(lattices) < best_fit_count:
        population = create_population(config, round((best_fit_count - len(lattices)) * 1.75))
        noisy_batch, lattice_batch, _ = generate_lattices(population.population.values(), config, noise_flag)
        lattices += lattice_batch
        print("New lattice batch generated, current population size:", len(lattices))
        if noise_flag:
            noisy += noisy_batch
    # np.save("Training_Carved.npy", np.asarray(lattices[:best_fit_count]))
    # np.save("Training_Carved_Noisy.npy", np.asarray(noisy[:best_fit_count]))
    return np.asarray(lattices[:best_fit_count]), noisy[:best_fit_count]


def create_population(config, pop_size=best_fit_count):
    """
    Generates a population of CPPN genomes according to the given CPPN-NEAT config file and population size.
    :param config: CPPN-NEAT config file specifying the parameters for the genomes.
    :param pop_size: Number of genomes to create.
    :return population: Population objecting containing a dictionary in the form {genome_id: genome_object}.
    """
    config.__setattr__("pop_size", pop_size)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    return population


def evaluate_half_fill(genome, config):
    """

    :param genome:
    :param config:
    :return:
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0
    lattice = np.zeros((20, 20, 20), dtype=bool)
    for (x, y, z) in value_range:
        if net.activate((x / 20, y / 20, z / 20))[0] >= 0.5:
            lattice[x][y][z] = True
    _, lattice = apply_constraints(lattice)
    for (x, y, z) in value_range:
        if lattice[x][y][z] != 0:
            fitness += 1
    if fitness > 4000:
        fitness = 4000 - fitness % 4001
    return fitness


def evaluate_half_fill_parallel(genomes, config):
    """

    :param genomes:
    :param config:
    :return:
    """
    evaluator = neat.parallel.ParallelEvaluator(thread_count, evaluate_half_fill)
    evaluator.evaluate(genomes, config)
    evaluator.__del__()
