"""Algorisme genètic d'AITK adaptat a paràmetres reals i llavors compartides."""

import copy
import contextlib
import io
import random

from aitk.algorithms import GeneticAlgorithm

FUZZY_BOUNDS = ((150.0, 800.0), (20.0, 220.0), (0.01, 0.45))


class Fitness:
    def __init__(self):
        self.values = ()

    @property
    def valid(self):
        return bool(self.values)


class Individual(list):
    def __init__(self, values):
        super().__init__(values)
        self.fitness = Fitness()


class RealGeneticAlgorithm(GeneticAlgorithm):
    """Usa la selecció i l'encreuament d'AITK amb mutacions reals acotades."""

    def __init__(self, kind, count):
        if kind == 'fuzzy':
            self.bounds = FUZZY_BOUNDS
            self.sigmas = (100, 30, 0.06)
            self.mutation_rate = 0.4
        elif kind == 'neural':
            self.bounds = [(-5.0, 5.0)] * count
            self.sigmas = [0.15] * count
            self.mutation_rate = 0.15
        else:
            raise ValueError(f'Tipus de controlador desconegut: {kind}')
        self.kind = kind
        # La mida es fixa en crear la població.
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__(len(self.bounds), 0)
        self.crossover_rate = 0.5
        self.mutation_rate = 0.4 if kind == 'fuzzy' else 0.15

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, value):
        self._population = value

    def make_random_chromosome(self):
        bounds = self.bounds if self.kind == 'fuzzy' else [(-1, 1)] * self.length
        return Individual(random.uniform(lo, hi) for lo, hi in bounds)

    def mutation(self, chromosome):
        for i, ((lo, hi), sigma) in enumerate(zip(self.bounds, self.sigmas)):
            if random.random() < self.mutation_rate:
                chromosome[i] = max(lo, min(hi, chromosome[i] + random.gauss(0, sigma)))


def make_toolbox(kind, count=0, *, prefix='Comparison'):
    return RealGeneticAlgorithm(kind, count)


def make_population(toolbox, n):
    toolbox.popSize = n
    toolbox.initialize_population()
    return toolbox.population


def next_population(population, toolbox):
    toolbox.population = population
    toolbox.popSize = len(population)
    toolbox.scores = [individual.fitness.values[0] for individual in population]
    toolbox.totalFitness = sum(toolbox.scores)
    # Conservem els dos millors també amb poblacions de mida imparella.
    elite_count = min(2, len(population))
    elites = [copy.deepcopy(individual) for individual in
              sorted(population, key=lambda item: item.fitness.values[0], reverse=True)[:elite_count]]
    offspring = []
    while len(offspring) < len(population) - elite_count:
        a, b = toolbox.crossover(toolbox.selection(), toolbox.selection())
        for child in (a, b):
            toolbox.mutation(child)
            offspring.append(Individual(child))
            if len(offspring) == len(population) - elite_count:
                break
    toolbox.generation += 1
    return elites + offspring


def seeds_for_trials(seed, trials, *, validation=False):
    start = seed + (10000 if validation else 0)
    return [start + i for i in range(trials)]
