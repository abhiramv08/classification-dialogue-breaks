import torch
import pandas as pd
import json
import re
import random
from transformers import BertTokenizer, BertForMaskedLM

device = "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)

def augment_text_by_mlm(text, mask_rate=0.3, max_predictions_per_seq=20):
    inputs = tokenizer.encode_plus(text, return_tensors='pt', max_length=512, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
   
    candidate_indices = [i for i, token in enumerate(tokens) if token not in ['[CLS]', '[SEP]']]
    
   
    n_tokens = len(candidate_indices)
    n_masked = min(max_predictions_per_seq, max(1, int(mask_rate * n_tokens)))

   
    n_masked = min(n_masked, len(candidate_indices))

    if n_masked == 0:
        return text

    masked_indices = random.sample(candidate_indices, n_masked)

    for index in masked_indices:
        tokens[index] = '[MASK]'

    masked_input_ids = tokenizer.convert_tokens_to_ids(tokens)
    inputs = torch.tensor([masked_input_ids]).to(device)
    with torch.no_grad():
        predictions = model(inputs)[0]

    for index in masked_indices:
        predicted_index = torch.argmax(predictions[0, index]).item()
        tokens[index] = tokenizer.convert_ids_to_tokens(predicted_index)
    
    augmented_text = tokenizer.convert_tokens_to_string(tokens).replace('[CLS] ', '').replace(' [SEP]', '')
    return augmented_text

def parse_key_value_pairs(input_string):
    parts = [part.strip() for part in re.split(r'(user|text|intent)', input_string) if part.strip()]
    result_dict = {}
    current_key = None
    for part in parts:
        if part in ['user', 'text', 'intent']:
            current_key = part
        else:
            cleaned_value = part.strip(" ,:;'\"")
            if current_key:
                result_dict[current_key] = cleaned_value
    return result_dict

def augment_row(row):
    u1 = row['utterance1']
    u2 = row['utterance2']
    
    u1 = parse_key_value_pairs(u1)
    u2 = parse_key_value_pairs(u2)
    
    print(f"Original utterance1: {u1}")
    print(f"Original utterance2: {u2}")
    
    u1['text'] = augment_text_by_mlm(u1['text'])
    u2['text'] = augment_text_by_mlm(u2['text'])

    row['utterance1'] = json.dumps(u1)
    row['utterance2'] = json.dumps(u2)
    
    print(f"Augmented utterance1: {u1}")
    print(f"Augmented utterance2: {u2}")
    return row

# Load the dataset
df = pd.read_csv('train.csv')

# Filter rows where label is 1
df_label1 = df[df['label'] == 1].copy()

# Apply augmentation to these rows
augmented_rows = df_label1.apply(augment_row, axis=1)

# Append the augmented rows to the original dataframe
df_augmented = pd.concat([df, augmented_rows])

# Save the augmented dataset
df_augmented.to_csv('train_augmented_mlm_bert.csv', index=False)
