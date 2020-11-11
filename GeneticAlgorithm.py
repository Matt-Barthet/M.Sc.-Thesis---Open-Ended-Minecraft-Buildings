from geneal.utils.helpers import get_elapsed_time
from Delenox_Config import generations_per_run
from geneal.genetic_algorithms.genetic_algorithm_base import GenAlgSolver
import numpy as np
import multiprocessing
import datetime
import logging


class GeneticAlgorithm(GenAlgSolver):

    def __init__(self, num_workers, variables_limits, number_of_runs, k, *args, **kwargs):
        GenAlgSolver.__init__(self, *args, **kwargs)
        self.variable_limits = variables_limits
        self.average_runs = number_of_runs
        self.num_workers = num_workers
        self.neighbors = k
        self.new_fitness = []
        self.new_population = []
        self.means = [[] for _ in range(generations_per_run)]
        self.std = [[] for _ in range(generations_per_run)]
        self.bests = [[] for _ in range(generations_per_run)]

    def solve(self):
        for run in range(self.average_runs):
            start_time = datetime.datetime.now()
            population = self.initialize_population()
            fitness = self.calculate_fitness(population)
            fitness, population = self.sort_by_fitness(fitness, population)
            gen_interval = max(round(self.max_gen / 10), 1)

            for gen_n in range(self.max_gen):

                if self.verbose and gen_n % gen_interval == 0:
                    logging.info(f"Iteration: {gen_n}")
                    logging.info(f"Best fitness: {fitness[0]}")

                self.means[gen_n].append(fitness.mean())
                self.bests[gen_n].append(fitness[0])

                ma, pa = self.select_parents(fitness)
                ix = np.arange(0, self.pop_size - self.pop_keep - 1, 2)
                xp = np.array(
                    list(map(lambda _: self.get_crossover_points(), range(self.n_matings)))
                )
                for i in range(xp.shape[0]):
                    population[-1 - ix[i], :] = self.create_offspring(
                        population[ma[i], :], population[pa[i], :], xp[i], "first"
                    )
                    population[-1 - ix[i] - 1, :] = self.create_offspring(
                        population[pa[i], :], population[ma[i], :], xp[i], "second"
                    )

                population = self.mutate_population(population, self.n_mutations)
                fitness = np.hstack((fitness[0], self.calculate_fitness(population[1:, :])))
                fitness, population = self.sort_by_fitness(fitness, population)

            self.generations_ = self.max_gen
            self.best_individual_ = population[0, :]
            self.best_fitness_ = fitness[0]
            self.population_ = population
            self.fitness_ = fitness

            if self.show_stats:
                end_time = datetime.datetime.now()
                time_str = get_elapsed_time(start_time, end_time)
                self.print_stats(time_str)

        means = []
        means_confidence = []
        bests = []
        bests_confidence = []

        for generation in range(self.max_gen):
            means.append(np.mean(self.means[generation]))
            means_confidence.append(np.std(self.means[generation]))
            bests.append(np.mean(self.bests[generation]))
            bests_confidence.append(np.std(self.bests[generation]))

        return means, means_confidence, bests, bests_confidence

    def calculate_fitness(self, population):
        self.new_population = population
        process_pool = multiprocessing.Pool(self.num_workers)
        fitness = process_pool.map(self.fitness_function, population)
        process_pool.close()
        process_pool.join()
        return np.asarray(fitness)

    def fitness_function(self, chromosome):
        distances = []
        for neighbour in range(len(self.new_population)):
            distance = 0
            for element in range(self.n_genes):
                distance += np.square(chromosome[element] - self.new_population[neighbour][element])
            distances.append(np.sqrt(distance))
        distances.sort()
        return np.round(np.average(distances[:self.neighbors]), 2)

    def initialize_population(self):
        return np.random.uniform(self.variable_limits[0], self.variable_limits[1], (self.pop_size, self.n_genes))

    def create_offspring(self, first_parent, sec_parent, crossover_pt, offspring_number):
        child = np.zeros(self.n_genes)
        for gene in range(self.n_genes):
            if gene < crossover_pt:
                child[gene] = first_parent[gene]
            else:
                child[gene] = sec_parent[gene]
        return child

    def mutate_population(self, population, n_mutations):
        for individual in population:
            for gene in range(self.n_genes):
                if np.random.random() < self.mutation_rate:
                    individual[gene] = np.random.uniform(self.variable_limits[0], self.variable_limits[1])
        return population
