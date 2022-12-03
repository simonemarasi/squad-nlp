from config import *
from bidafLike.train import train_bidaf_like
from bidafLike.test import test_bidaf_like

def bidaf_runner(filepath, outputdir=BIDAF_WEIGHTS_PATH, mode="test", load_embedding=True):

    print("###########################")
    print("#### BIDAF-LIKE RUNNER ####")
    print("###########################")

    if mode == 'train':
        train_bidaf_like(filepath, load_embedding, outputdir)
    elif mode == 'test':
        test_bidaf_like(filepath, outputdir)
