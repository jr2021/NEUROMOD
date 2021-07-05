import pandas as pd
import numpy as np
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.ensemble import *
from sklearn.neural_network import *
import sys
import math
import json
import pickle
import IPython as ipy

class NEUROMOD():

    population_size = 200
    parents_size = 100
    children_size = 100
    max_generations = 500

    def __init__(self, train_file, test_file, validate_file, N=50, thresh=1):
        self.features = int(features)
        self.train = pd.read_csv(train_file, index_col=0).values
        self.test = pd.read_csv(test_file, index_col=0).values
        self.validate = pd.read_csv(validate_file, index_col=0).values
        self.thresh, self.N = float(thresh), int(N)
        self.max_crowding = []


    def genetic_algorithm(self):

        population = self.initialize()
        population = self.evaluate(population)
        for generation in range(self.max_generations):
            parents, fronts = self.nsga_ii(population, self.parents_size)
            children = self.recombine(parents)
            children = self.mutate(children)
            children = self.evaluate(children)
            population, fronts = self.nsga_ii(np.concatenate((population, children)), self.population_size)

            self.max_crowding.append(np.max([individual['meta']['distance'] for individual in population if individual['meta']['distance'] < math.inf]))
            self.display_dynamic()

            if self.crowding_stagnation(max_crowding, generation):
                break

        self.pareto = fronts[0]
        self.display_static()

    # CLI display

    def display_dynamic(self):
        pass # FIXME

    def display_static(self):
        pass # FIXME

    # population initialization

    def initialize(self):
        return [{'meta': {'test': None,
                          'validate': None,
                          'dominates': None,
                          'dominated': None,
                          'distance': None},
                 'data': []} for _ in range(self.population_size)]

    # fitness evaluation

    def evaluate(self, population):

        for individual in population:
            pass # FIXME

        return population


    def accuracy(self, data, test):
        X_test, y_test = test[:,1:], test[:,0]

        pass # FIXME

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
                if population[i]['meta']['test'] > population[j]['meta']['test'] and population[i]['meta']['size'] < population[j]['meta']['size']:
                    population[i]['meta']['dominates'].add(j)
                if population[j]['meta']['test'] > population[i]['meta']['test'] and population[j]['meta']['size'] < population[i]['meta']['size']:
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
        objectives = [] # FIXME

        for front in fronts:
            if len(front) > 0:
                for objective in objectives:
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
        child = {'meta': {'test': None,
                          'validate': None,
                          'size': None,
                          'dominates': None,
                          'dominated': None,
                          'distance': None},
                 'index': self.index,
                 'data': None}

        # FIXME

        return child

    # mutation

    def mutate(self, children):

        for i in range(self.children_size):

            j = np.random.randint(len(self.index))

            if children[i]['data'][j] == 1:
                if np.sum(children[i]['data']) > 1:
                    children[i]['data'][j] = 0
            else:
                children[i]['data'][j] = 1

        return children

    # termination

    def crowding_stagnation(self, max_crowding, generation):

        std = np.array(max_crowding[-self.N:]).std()
        return std < self.thresh and generation > self.N
