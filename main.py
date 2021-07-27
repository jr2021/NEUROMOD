from NEUROMOD import NEUROMOD
import sys
import pickle
import datetime

def main():
    instance = NEUROMOD(sys.argv[1], sys.argv[2])
    instance.genetic_algorithm()
    with open(str(datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")) + '.pkl', 'wb') as file:
        pickle.dump(instance, file)

main()
