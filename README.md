# classification-dialogue-breaks

# Introduction
This project implements a text classification task on the GUIDE(Growing Understanding through Interactive Daily Experienc) dataset. The dataset consists of simulated interactions between two characters in home settings, created using the VirtualHome Multi-Agent Household Simulator. 

The classification task is to detect whether there is a shift between a pair of utterances by the two characters. 
In this implementation, we have used two large language models, BERT and RoBERTa, for this task. The base implementations use the json formatted utterances, containing the dialogue, the user and the intent. The bert_intent and roberta_intent implementations use the intent categories as a separate feature in addition to the text data for classification.

Reference for the transformers library used: - [huggingface transformers library](https://huggingface.co/transformers/v2.2.0/index.html)

# Run and Evaluation
The `augmented_dataset` folder contains the augmented datasets, along with the code used.
We have used two techniques: paraphrasing and LLMs to augment the data and make it balanced in terms of output labels.

The models are present in the `models` folder
Below is an example run command:

```
 python models/roberta.py --train_data='train.csv' --test_data='test.csv' --epochs=10
```

The results are shared in the `results` folder. Below is the result from the roberta base model:

```
{
    "Segment Continuation": {
        "precision": 0.8975501113585747,
        "recall": 0.9005586592178771,
        "f1-score": 0.8990518683770218,
        "support": 895.0
    },
    "Segment Shift": {
        "precision": 0.7729591836734694,
        "recall": 0.7670886075949367,
        "f1-score": 0.770012706480305,
        "support": 395.0
    },
    "accuracy": 0.8596899224806202,
    "macro avg": {
        "precision": 0.8352546475160221,
        "recall": 0.8338236334064069,
        "f1-score": 0.8345322874286634,
        "support": 1290.0
    },
    "weighted avg": {
        "precision": 0.8594001761371665,
        "recall": 0.8596899224806202,
        "f1-score": 0.8595398769435311,
        "support": 1290.0
    }
}
```