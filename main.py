from argparse import ArgumentParser
from config import DATA_PATH

parser = ArgumentParser()
parser.add_argument("-mode",
                    dest="mode",
                    required=True,
                    choices=['train', 'test', 'evaluate'],
                    default='test',
                    help="Select the mode to run the model in",
                    metavar="MODE")    

parser.add_argument("-df", "--data-file",
                    dest="datafile",
                    default=DATA_PATH,
                    help="Name of the .json file used for training or testing",
                    metavar="DATA_FILES")

parser.add_argument("-we", "--word-embeddings",
                    dest="embeddings",
                    help="Don't use pre-generated embedding model and generate embedding matrix from scratch (only for GloVe models)",
                    action='store_false')

parser.add_argument("-wd", "--weights-directory", 
                    dest="weightsdir",
                    help="Name of the directory where save or load the weights of the model (depends on the mode used)",
                    metavar="WEIGHTS_DIRECTORY")

parser.add_argument("-od", "--output-directory", 
                    dest="outputdir",
                    help="Name of the directory where to output the prediction file (test mode only)",
                    metavar="OUTPUT_DIRECTORY")

parser.add_argument("-pred", "--prediction_file",
                    dest="prediction_file",
                    help=".txt file generated after the testing stage", 
                    metavar="PREDICTION_FILE")


if __name__ == '__main__':

    args = vars(parser.parse_args())

    print("######################")
    print("#### SQuAd RUNNER ####")
    print("######################")

    if (args['mode'] == 'evaluate'):
        from evaluate_simple import evaluate_predictions
        evaluate_predictions(args["datafile"], args["prediction_file"])
        input("Press any key to terminate")
        exit()

    print("\nModels available:\n")
    print("1) GloVe")
    print("2) BERT")

    while True:
        model_to_run = input("\nChoose the model you want to run: ")
        if model_to_run not in ["1", "2"]:
            print("Error, please make your choice between the ones allowed")
            continue
        else:
            break
        
    if model_to_run == "1":
        from glove_runner import glove_runner
        glove_runner(args['datafile'], args['outputdir'], args['weightsdir'], args['mode'], args['embeddings'])
    elif model_to_run == "2":
        from bert_runner import bert_runner
        bert_runner(args['datafile'], args['outputdir'], args['weightsdir'], args['mode'])
