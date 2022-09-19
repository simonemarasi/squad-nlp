from argparse import ArgumentParser
from config import *
from glove_runner import glove_runner
from bert_runner import bert_runner
from bidaf_runner import bidaf_runner

parser = ArgumentParser()
parser.add_argument("-mode", 
                    dest="mode", 
                    required=True,
                    choices=['train', 'test'],
                    help="Select the mode to run the model in.",
                    metavar="MODE")    

parser.add_argument("-df", "--data-file", 
                    dest="datafile", 
                    default=DATA_PATH,
                    help="Name of the json file for training or testing",
                    metavar="DIR")

parser.add_argument("-od", "--output-directory", 
                    dest="outputdir", 
                    #default=GLOVE_WEIGHTS_PATH,
                    help="Name of the directory where the output from training is saved (weights and history)",
                    metavar="DIR")

parser.add_argument("-wf", "--weights_file",
                    dest="weights",
                    required=False,
                    help=".h5 file where the model weights are saved. Loaded to continue training or testing", metavar="weightfile.h5")

args = vars(parser.parse_args())

if __name__ == '__main__':

    print("######################")
    print("#### SQuAd RUNNER ####")
    print("######################")
    print("\nModels available:\n")
    print("1) GloVe")
    print("2) BERT")
    print("3) Bidaf")

    while True:
        model_to_run = input("\nChoose the model you want to run: ")
        if model_to_run not in ["1", "2", "3"]:
            print("Error, please make your choice between the ones allowed")
            continue
        else:
            break
    if model_to_run == "1":
        model = glove_runner(args['datafile'], args['outputdir'])
    elif model_to_run == "2":
        model = bert_runner(args['datafile'], args['outputdir'])
    elif model_to_run == "3":
        model = bidaf_runner(args['datafile'], args['outputdir'])
    pass
