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

def glove_runner(filepath, outputdir, weightsdir, mode, generate_embeddings):

    weightsdir = GLOVE_WEIGHTS_PATH if weightsdir is None else weightsdir
    load_embeddings = generate_embeddings if generate_embeddings is False else True

    print("######################")
    print("#### GLOVE RUNNER ####")
    print("######################")
    print("\nModels available:\n")
    print("1) Baseline")
    print("2) Baseline with attention")
    print("3) Baseline with features")
    print("4) Baseline with char embeddings and attention")
    
    while True:
        model_choice = get_model_input("\nPlease type the number of the variant of the GloVe model you want to run:")
        if model_choice not in ["1", "2", "3", "4"]:
            print("Error, please make your choice between the ones allowed")
            continue
        else:
            break

    if mode == 'train':
        print("\nRunning GloVe model in train mode\n")
        train_glove(filepath, load_embeddings, model_choice, weightsdir)
    elif mode == 'test':
        print("\nRunning GloVe model in test mode\n")
        test_glove(filepath, model_choice, outputdir, weightsdir)