# Hate-Speech-Detection
In this experiment, our focus was on studying how different models can effectively detect hate speech across different domains. We conduct an experiment involving five domains, where model is trained on the data from one source and then evaluated on the test data from the other four datasets along with Hate Check dataset.

## General information
In all of our experiments, we utilized [`HateBERT`](https://huggingface.co/GroNLP/hateBERT), an English re-trained BERT model specifically designed for detecting abusive language.
All code makes use of Huggingface models and the Transformers package to import these models.

## Requirements
Tested on Python 3.11.3, but anything close to that is likely to work as well.

To run the model, one is required to install a small number of packages which can be found in requirements.txt.
