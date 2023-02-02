from glove.train import train_glove
from glove.test import test_glove
from config import GLOVE_WEIGHTS_PATH

def get_model_input(prompt):
    while True:
        value = input(prompt)
        if value not in ["1", "2", "3", "4"]:
            print("Sorry, your choice must be between the four allowed")
            continue
        else:
            break
    return value

def glove_runner(filepath, outputdir=None, weightsdir=GLOVE_WEIGHTS_PATH, mode="test", load_embedding=True):

    print("######################")
    print("#### GLOVE RUNNER ####")
    print("######################")
    print("\nModels available:\n")
    print("1) Baseline")
    print("2) Baseline with attention")
    print("3) Baseline with features")
    print("4) Baseline with attention and features")
    model_choice = get_model_input("\nPlease type the model number to run with the current configuration: ")

    if mode == 'train':
        print("\nRunning GloVe model in train mode\n")
        train_glove(filepath, load_embedding, model_choice, weightsdir)
    elif mode == 'test':
        print("\nRunning GloVe model in test mode\n")
        test_glove(filepath, model_choice, outputdir, weightsdir)