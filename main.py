from NEUROMOD import NEUROMOD
import sys
import pickle
import datetime

def main():
    instance = MRGA(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.arg$
    instance.run()
    with open(str(datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")) + '.pkl'$
        pickle.dump(instance, file)

main()
