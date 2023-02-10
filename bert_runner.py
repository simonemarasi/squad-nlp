from config import BERT_WEIGHTS_PATH 

def get_model_input(prompt):
    while True:
        value = input(prompt)
        if value not in ["1", "2", "3"]:
            print("Sorry, your choice must be between the three allowed")
            continue
        else:
            break
    return value

def bert_runner(filepath, outputdir, weightsdir, mode):

    weightsdir = BERT_WEIGHTS_PATH if weightsdir is None else weightsdir

    print("#####################")
    print("#### BERT RUNNER ####")
    print("#####################")
    print("\nModels available:\n")
    print("1) Baseline")
    print("2) Baseline with RNN")
    print("3) Baseline with RNN and features")

    while True:
        model_choice = get_model_input("\nPlease type the number of the variant of the BERT model you want to run: ")
        if model_choice not in ["1", "2", "3"]:
            print("Error, please make your choice between the ones allowed")
            continue
        else:
            break

    if mode == 'train':
        from bert.train import train_bert
        print("\nRunning BERT model in train mode\n")
        train_bert(filepath, model_choice, weightsdir)
    elif mode == 'test':
        from bert.test import test_bert
        print("\nRunning BERT model in test mode\n")
        test_bert(filepath, model_choice, outputdir, weightsdir)