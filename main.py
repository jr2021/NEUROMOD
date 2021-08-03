from NEUROMOD import NEUROMOD
import pickle
import datetime
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", "-l", type=int, help="generations to look back", default=50)
    parser.add_argument("--theta", "-t", type=float, help="threshold for std of maximum crowding distance", default=1.0)
    parser.add_argument("--population_size", type=int, help="population size", default=20)
    parser.add_argument("--parents_size", type=int, help="parent population size", default=10)
    parser.add_argument("--children_size", type=int, help="children population size", default=10)
    parser.add_argument("--max_generations", type=int, help="max generations", default=1000)
    parser.add_argument("--data", "-d", type=str, help="dataset to evaluate on", default='MNIST')
    parser.add_argument("--train_set_size", type=int, help="number of samples to use in training set", default=200)
    parser.add_argument("--batch_size", type=int, help="number of samples per batch", default=200)
    parser.add_argument("--phys", "-p", type=str, help="physical constraint", default='count')
    parser.add_argument("--individual_mutation", type=float, help="individual mutation rate", default=0.5)
    parser.add_argument("--layerwise_mutation", type=float, help="layer-wise mutation rate", default=0.5)
    parser.add_argument("--density", type=float, help="density of nonzero weights", default=0.7)
    args = parser.parse_args()


    instance = NEUROMOD(n=args.lookback,
                        theta=args.theta,
                        population_size = args.population_size,
                        parents_size = args.parents_size,
                        children_size = args.children_size,
                        max_generations = args.max_generations,
                        data=args.data, phys = args.phys,
                        train_set_size = args.train_set_size, batch_size = args.batch_size,
                        individual_mutation = args.individual_mutation,
                        layerwise_mutation = args.layerwise_mutation,
                        density = args.density)
    instance.genetic_algorithm()
    with open(str(datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")) + '.pkl', 'wb') as file:
        pickle.dump(instance, file)

if __name__ == "__main__":
    main()
