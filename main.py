from NEUROMOD import NEUROMOD
import sys
import pickle
import datetime

def main():
    instance = NEUROMOD(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    instance.genetic_algorithm()
    with open('test.pkl', 'wb') as file:
        pickle.dump(instance, file)

main()
