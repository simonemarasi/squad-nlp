# SQuAD Question Answering
## UniBo - NLP Exam - Artificial Intelligence

### Introduction
This repository contains the implementation of a form of extractive QA applied to the Stanford Question Answering Dataset (SQuAD). Given a question and a passage containing the answer, the task is to predict the answer text span in the passage. In particular it has been tried several models
- a very simple Recurrent Neural Network (RNN) using the GloVe embedding;
- an attention based model using both GloVe and a pretrained Character Level embedding, 
- an high end reference model using BERT


### Run script
First of all, you have to clone the repository into your PC through the usual git command:

`https://github.com/simonesimo97/squad-nlp.git`

To execute the script in a test fashion with pretrained word embeddings and model weights [download the .zip file](http) and extract it into the `data/models` folder. After doing that your root folder structure should be like the following:

```
bert
└── ...
common
└── ...
data
├── training_set.json
├── ag-news
│   ├── train.csv
│   └── test.csv
└── models
    ├── bert
    │   └── weights
    │       ├── baseline
    │       │   └── weights.h5
    │       ├── rnn
    │       │   └── weights.h5
    │       └── rnn-features
    │           └── weights.h5
    └── glove
        ├── embedding_model.pkl
        └── weights
            ├── baseline
            │   ├── weights.data-00000-of-00001
            │   └── weights.index
            ├── attention
            │   ├── weights.data-00000-of-00001
            │   └── weights.index
            ├── features
            │   ├── weights.data-00000-of-00001
            │   └── weights.index
            └── char
                ├── weights.data-00000-of-00001
                ├── weights.index
                ├── CNN_100_FineTunedEmbedding.data-00000-of-00001
                └── CNN_100_FineTunedEmbedding.index
glove
└── ...
```

Then from your terminal you can launch the `main.py` file and choose the model you want to run and its variant. You can also specify other parameters such as the mode in which run (train, test or evaluate), the custom weights directory where to pick weights file (if one), output directory for predictions (if test mode), if to use the pretrained GloVe embeddings or build it from scratch and so on. You can see all available options running the help command:

`help command`

After the execution of the script, if you have chosen the **train mode** then you have the just computed weight file in the output directory specified and you are now ready to test on different data file to see how it performs. 

If you have chosen instead the **test mode** on your dataset now you have a `prediction.txt` file with the results in the output directory specified. 

You can relaunch the script in the **evaluate mode** specifying the paths of the prediction file and of the data file in order to get the metrics of that model. These metrics are computed using the SQuAD official evaluation script.

### Results on official SQuAD test set

| Model                          	| Exact Match (%) 	| F1 score (%) 	|
|--------------------------------	|-----------------	|--------------	|
| GloVe Baseline                 	| 6.55            	| 10.55        	|
| GloVe Features                 	| 34.70           	| 45.68        	|
| GloVe Attention                	| 51.60           	| 64.03        	|
| **Glove Attention + Char RNN** 	| **53.16**       	| **66.01**    	|
| BERT Baseline                  	| 71.19           	| 79.79        	|
| **BERT RNN**                   	| **74.23**       	| **82.47**    	|
| BERT RNN + Features            	| 71.95           	| 81.01        	|

### Examples

### Resources

You can download weights file even separately but be sure to respect the folder structure as depicted before, possibly creating by yourself missing folders.

Only models weights files: [Download]()

Pre-trained Char RNN weights file: [Download]()

GloVe embedding model: [Download](https://www.4sync.com/web/directDownload/1fLGKUVR/GELocHMl.14ab03219e9ec9989ecb72c5a99ed420)