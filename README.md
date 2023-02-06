# SQuAD Question Answering
## UniBo - NLP Exam - Artificial Intelligence

### Introduction
This repository contains the implementation of a form of extractive QA applied to the Stanford Question Answering Dataset (SQuAD). Given a question and a passage containing the answer, the task is to predict the answer text span in the passage. In particular we have tried several models
- a very simple Recurrent Neural Network (RNN) using the GloVe embedding;
- an attention based model using both GloVe and a pretrained Character Level embedding, 
- a high end reference model using BERT


### Setup
First of all, you need to clone the repository into your PC through the usual git command:

`git clone https://github.com/simonesimo97/squad-nlp.git`

To execute the script in a test fashion with pretrained word embeddings and model weights [download the .zip file](https://www.4sync.com/web/directDownload/vq1HmCVf/GELocHMl.3efe1a6ed6f7215faddb42bf203a0904) and extract it into the `data/models` folder. After doing that your root folder structure should be like the following:

<pre>
bert
└── ...
common
└── ...
data
├── training_set.json
├── ag-news
│   ├── train.csv
│   └── test.csv
<b>└── models
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
                └── CNN_100_FineTunedEmbedding.index</b>
glove
└── ...
...
</pre>

### Script Execution Modes

* **Train:** After the execution of the script you will find the just trained weight file in the output directory specified and you are now ready to test the model on a different data file to see how it performs.
* **Test:** After execution you will find a `prediction.txt` file with the results in the output directory specified. 
* **Evaluate:** After execution, specifying the paths of the prediction file and of the target data file, the script will print to console the metrics of that model. These metrics are computed using the SQuAD official evaluation script.

### Run

All library requirements are reported in the `requirements.txt` file. You can install running the command `pip install -r requirements.txt`. We strongly suggest to install the required packages in a separate virtual environment.

Then from your terminal you can launch the `main.py` file and choose the model you want to run and its variant. You can also specify other parameters such as the mode in which to run (train, test or evaluate), a custom weights directory from where to pull the model weights, output directory for predictions (if in test mode), if to use the pretrained GloVe embeddings or build them from scratch and so on. Here are some examples:

`py main.py -mode=train [-we] [-wd=/path/where/saving/weights]`

`py main.py -mode=test -df=/path/of/test/set [-we] [-wd=/path/where/saving/weights] [-od="/path/where/saving/predictions]"`

`py main.py -mode=evaluate -df=/path/of/test/set -pred="/path/of/predictions/file"`

The variables written above inside the square brackets have a default value you can check in the `config.py` file that you can find in the root folder of the repository.

### Results on official SQuAD Test Set

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

![Qualitative error analysis of some question-answer couples](ErrorAnalysisResults.jpg)

### Resources

If you want to train the models autonomously, you can download the Char RNN weights file or the GloVe embedding model files separately but be sure to respect the folder structure as depicted before, possibly creating by yourself missing folders.

Pre-trained Char RNN weights file: [Download](https://www.4sync.com/web/directDownload/Kun91r2F/GELocHMl.ba205b4c346baa151eb66c73fc4f7853)

GloVe embedding model: [Download](https://www.4sync.com/web/directDownload/1fLGKUVR/GELocHMl.14ab03219e9ec9989ecb72c5a99ed420)