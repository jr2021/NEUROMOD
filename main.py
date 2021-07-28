from NEUROMOD import NEUROMOD
import sys
import pickle
import datetime
import argparse

'''
arg1: number of generations to look back
arg2: theta (threshold for std of maximum crowding distance)
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", "-l", type=int, help="generations to look back", default=50)
    parser.add_argument("--theta", "-t", type=float, help="threshold for std of maximum crowding distance", default=1.0)
    parser.add_argument("--population_size", type=int, help="population size", default=20)
    parser.add_argument("--parents_size", type=int, help="parent population size", default=10)
    parser.add_argument("--children_size", type=int, help="children population size", default=10)
    parser.add_argument("--max_generations", type=int, help="max generations", default=500)
    args = parser.parse_args()


    instance = NEUROMOD(n=args.lookback, theta=args.theta, population_size = args.population_size, parents_size = args.parents_size, children_size = args.children_size, max_generations = args.max_generations)
    instance.genetic_algorithm()
    with open(str(datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")) + '.pkl', 'wb') as file:
        pickle.dump(instance, file)

if __name__ == "__main__":
    main()
