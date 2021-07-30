import pandas as pd
import numpy as np
from fitness import evaluate_fitness, DatasetObjective, NetworkCostObjective
import math
import time
import warnings
warnings.filterwarnings('ignore', message = 'The given NumPy array is not writeable')

class NEUROMOD():

    def __init__(self, n=50, theta=1, population_size=20, parents_size=10, children_size=10, max_generations=500, data='MNIST', phys='l1', individual_mutation=0.5, layerwise_mutation=0.5):
        self.theta, self.n = float(theta), int(n)
        self.population_size, self.parents_size, self.children_size, self.max_generations = int(population_size), int(parents_size), int(children_size), int(max_generations)
        self.individual_mutation, self.layerwise_mutation = float(individual_mutation), float(layerwise_mutation)
        self.weight_sizes = [(28*28, 100), (100,), (100, 10), (10,)]

        self.objectives = {'acc': DatasetObjective(dataset_name = 'MNIST', evaluation_metric = 'acc'),
                           'phys': {'count': NetworkCostObjective(cost_metric = 'count'),
                                    'l1': NetworkCostObjective(cost_metric = 'l1'),
                                    'l2': NetworkCostObjective(cost_metric = 'l2')}[phys]}
        self.statistics = {'fitness' : {'all': {'acc': [], 'val': [], 'phys': []},
                                        'avg': {'acc': [], 'val': [], 'phys': []},
                                        'min': {'acc': [], 'val': [], 'phys': []},
                                        'max': {'acc': [], 'val': [], 'phys': []}},
                           'pareto': {'crowd': {'max': [], 'std': []}}}


    def genetic_algorithm(self):

        print('Starting evolution with objectives:', list(self.objectives.values()))
        start = time.time()

        population = self.initialize()
        population = self.evaluate(population)
        for generation in range(self.max_generations):
            print(f'Generation {generation+1}/{self.max_generations} | Elapsed time: {time.time() - start:.2f}')
            parents, fronts = self.nsga_ii(population, self.parents_size)
            children = self.recombine(parents)
            children = self.mutate(children)
            children = self.evaluate(children)

            self.update_dynamic(np.concatenate((population, children)))
            self.display_dynamic()

            population, fronts = self.nsga_ii(np.concatenate((population, children)), self.population_size)

            self.statistics['pareto']['crowd']['max'].append(np.max([individual['meta']['distance'] for individual in population if individual['meta']['distance'] < math.inf]))
            
            if self.crowding_stagnation(generation):
                break

        self.pareto = fronts[0]
        self.update_static()
        self.display_static()

    # CLI display

    def display_dynamic(self):
        print('std:', self.statistics['pareto']['crowd']['std'][-1:])
        print('mean acc:', np.mean(self.statistics['fitness']['all']['acc']), 'mean network cost:', np.mean(self.statistics['fitness']['all']['phys']))

    def display_static(self):
        print('mean acc:', self.statistics['fitness']['avg']['acc'][-1:], 'mean network cost:', self.statistics['fitness']['avg']['phys'][-1:])

    # population initialization
    def init_genome(self, sizes, p = 0.5):
        """
        Initalizes the weights using randn, with probability p of being 0
        
        Input
        sizes: list of size tuples
        
        Returns: list of numpy arrays (to become PyTorch tensors)
        """
        return [np.random.randn(*s) * np.random.binomial(n=1, p=p, size = s) for s in sizes]

    def initialize(self):
        return [{'meta': {'acc': None,
                          'val': 0,
                          'phys': None,
                          'dominates': None,
                          'dominated': None,
                          'distance': None},
                 'data': self.init_genome(self.weight_sizes)} for _ in range(self.population_size)]

    # fitness evaluation

    def evaluate(self, population):

        for individual_number, individual in enumerate(population):
            individual['meta']['acc'] = evaluate_fitness(individual['data'], self.objectives['acc'])
            individual['meta']['phys'] = evaluate_fitness(individual['data'], self.objectives['phys'])

        return population


    # parent/survivor selection

    def nsga_ii(self, population, selection_size):

        selection = []
        j, k = 0, 0

        fronts = self.fast_nondominated_sort(population, selection_size)
        fronts = self.crowding_distance_assignment(fronts)

        while len(selection) + len(fronts[j]) < selection_size:
            for individual in fronts[j]:
                selection.append(individual)

            j += 1

        fronts[j] = sorted(fronts[j], key=lambda individual: individual['meta']['distance'], reverse=True)

        while len(selection) < selection_size:
            selection.append(fronts[j][k])
            k += 1

        return selection, fronts


    def fast_nondominated_sort(self, population, num_fronts):
        fronts, k = [[] for _ in range(num_fronts)], 0

        for i in range(len(population)):
            population[i]['meta']['dominates'], population[i]['meta']['dominated'], population[i]['meta']['distance'] = set(), 0, 0
            for j in range(len(population)):
                if population[i]['meta']['acc'] > population[j]['meta']['acc'] and population[i]['meta']['phys'] < population[j]['meta']['phys']:
                    population[i]['meta']['dominates'].add(j)
                if population[j]['meta']['acc'] > population[i]['meta']['acc'] and population[j]['meta']['phys'] < population[i]['meta']['phys']:
                    population[i]['meta']['dominated'] += 1
            if population[i]['meta']['dominated'] == 0:
                fronts[0].append(population[i])

        while len(fronts[k]) > 0:
            for i in range(len(fronts[k])):
                for j in fronts[k][i]['meta']['dominates']:
                    population[j]['meta']['dominated'] -= 1
                    if population[j]['meta']['dominated'] == 0:
                        fronts[k + 1].append(population[j])
            k += 1

        return fronts


    def crowding_distance_assignment(self, fronts):

        for front in fronts:
            if len(front) > 0:
                for objective in self.objectives.keys():
                    front = sorted(front, key=lambda individual: individual['meta'][objective])
                    front[0]['meta']['distance'], front[-1]['meta']['distance'] = math.inf, math.inf
                    for i in range(2, len(front) - 1):
                        front[i]['meta']['distance'] += front[i + 1]['meta'][objective] - front[i - 1]['meta'][objective]

        return fronts


    # recombination

    def recombine(self, parents):
        children = []

        np.random.shuffle(parents)

        for i in range(len(parents) - 1):
            children.append(self.one_point(parents[i], parents[i + 1]))

        children.append(self.one_point(parents[0], parents[-1]))

        return children


    def one_point(self, mother, father):
        child = {'meta': {'acc': None,
                          'val': 0,
                          'phys': None,
                          'dominates': None,
                          'dominated': None,
                          'distance': None},
                 'data': mother['data']}

        return child

    # mutation

    def mutate(self, children):

        for i in range(self.children_size):
            if np.random.uniform() > self.individual_mutation:
                for j in range(len(children[i]['data'])):
                    if np.random.uniform() > self.layerwise_mutation:
                        layer = np.hstack(children[i]['data'][j])
                        n = np.random.randint(layer.size-1)
                        index = np.random.randint(layer.size, size=n)
                        layer[index] = np.random.randn(n) * np.random.binomial(1, p=0.5, size=n)
                        children[i]['data'][j] = layer.reshape(children[i]['data'][j].shape)

        return children

    # termination

    def crowding_stagnation(self, generation):
        std = np.array(self.statistics['pareto']['crowd']['max'][-self.n:]).std()
        self.statistics['pareto']['crowd']['std'].append(std)
        return std < self.theta and generation > self.n


    def update_dynamic(self, population):
        self.statistics['fitness']['all']['acc'].append([individual['meta']['acc'] for individual in population])
        self.statistics['fitness']['all']['val'].append([individual['meta']['val'] for individual in population])
        self.statistics['fitness']['all']['phys'].append([individual['meta']['phys'] for individual in population])


    def update_static(self):
        self.statistics['fitness']['min']['acc'] = [np.min(generation) for generation in self.statistics['fitness']['all']['acc']]
        self.statistics['fitness']['avg']['acc'] = [np.mean(generation) for generation in self.statistics['fitness']['all']['acc']]
        self.statistics['fitness']['max']['acc'] = [np.max(generation) for generation in self.statistics['fitness']['all']['acc']]

        self.statistics['fitness']['min']['val'] = [np.min(generation) for generation in self.statistics['fitness']['all']['val']]
        self.statistics['fitness']['avg']['val'] = [np.mean(generation) for generation in self.statistics['fitness']['all']['val']]
        self.statistics['fitness']['max']['val'] = [np.max(generation) for generation in self.statistics['fitness']['all']['val']]

        self.statistics['fitness']['min']['phys'] = [np.min(generation) for generation in self.statistics['fitness']['all']['phys']]
        self.statistics['fitness']['avg']['phys'] = [np.mean(generation) for generation in self.statistics['fitness']['all']['phys']]
        self.statistics['fitness']['max']['phys'] = [np.max(generation) for generation in self.statistics['fitness']['all']['phys']]
