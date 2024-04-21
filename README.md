# classification-dialogue-breaks

# Introduction
This project implements a text classification task on the GUIDE(Growing Understanding through Interactive Daily Experienc) dataset. The dataset consists of simulated interactions between two characters in home settings, created using the VirtualHome Multi-Agent Household Simulator. 

The classification task is to detect whether there is a shift between a pair of utterances by the two characters. 
In this implementation, we have used two large language models, BERT and RoBERTa, for this task. The base implementations use the json formatted utterances, containing the dialogue, the user and the intent. The bert_intent and roberta_intent implementations use the intent categories as a separate feature in addition to the text data for classification.

Reference for the transformers library used: - [huggingface transformers library](https://huggingface.co/transformers/v2.2.0/index.html)

# Run and Evaluation
The models are present in the `models` folder
Below is an example run command:

```
 python models/roberta.py --train_data='train.csv' --test_data='test.csv' --epochs=10
```

The results are shared in the `results` folder. Below is the result from the roberta base model:

![RoBERTa result](roberta_results.png)