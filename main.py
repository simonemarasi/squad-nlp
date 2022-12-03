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
                    help="Select the mode to run the model in",
                    metavar="MODE")    

parser.add_argument("-df", "--data-file",
                    dest="datafile",
                    default=DATA_PATH,
                    help="Name of the .json file used for training or testing",
                    metavar="DATA_FILES")

parser.add_argument("-emb", "--embedding",
                    dest="embedding",
                    default=True,
                    choices=[True, False],
                    required=False,
                    help="Whether or not use pre-generated embedding model (only for GloVe and Bidaf-Like models)",
                    metavar="EMBEDDING")

parser.add_argument("-od", "--output-directory", 
                    dest="outputdir", 
                    #default=GLOVE_WEIGHTS_PATH,
                    help="Name of the directory where the output from training is saved (weights and history)",
                    metavar="OUTPUT_DIRECTORY")

parser.add_argument("-wf", "--weights_file",
                    dest="weights",
                    required=False,
                    help=".h5 file where the model weights are saved. Loaded to continue training or testing", 
                    metavar="WEIGHTS_FILE")

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
        glove_runner(args['datafile'], args['outputdir'], args['mode'], args['embedding'])
    elif model_to_run == "2":
        bert_runner(args['datafile'], args['outputdir'], args['mode'])
    elif model_to_run == "3":
        bidaf_runner(args['datafile'], args['outputdir'], args['mode'], args['embedding'])
    
