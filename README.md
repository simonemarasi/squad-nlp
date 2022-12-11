# SQuAD Question Answering
## UniBo - NLP Exam - Artificial Intelligence

### Introduction
This repository contains several implementation of model used to perform a question-answering task over the SQuAD Dataset. In particular it has been tried several models and variants of them. For simplicity it is possible to divide them in three macrocategories:
- **GloVe** based
- **BERT** based

Moreover the variants implemented have been taken into account additional features, char embeddings, deep recurrent layers and attention implementations.

#### GloVe
TODO

#### BERT
TODO

### Run scripts
You have to clone the repository into your PC through the usual git command:

`git clone ......`

To execute the script in a test fashion pretrained word embeddings and char neural network weights have to be preliminarily dowloaded from the following [link](link here "Download") and placed in the correct path, as in the following picture:

![](https://pandao.github.io/editor.md/images/logos/editormd-logo-180x180.png)

Then from your terminal you can launch the `main.py` file and choose model and the eventual variant. You can also specify other parameters such as the mode in which run (train or test), weights directory where to pick weights file, output directory for predictions (if test mode), if use the pretrained GloVe embeddings or build it from scratch and so on. You can see all available option with the help command:

`help command`

After the execution of the script, if you have chosen the **train mode** then you have the yet computed weight file in the output directory specified and you are now ready to test on different data to see how it performs. 

If you have chosen instead the **test mode** on your dataset now you have a prediction.txt file in the output directory specified. 

You can relaunch the script in the **evaluate mode** specifying the path of the prediction file and the file of the dataset you chosen to generate it in order to have the metrics of the model obtained on that data. These metrics are computed using the SQuAD official evaluation script.

### Examples

### Resources
All model weights files: [Download]()
Char weights file: [Download]()
GloVe embedding model: [Download]()