import torch
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase_text(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

def parse_key_value_pairs(input_string):
    # Split the string based on the keys and drop the first empty split from the leading split
    parts = [part.strip() for part in re.split(r'(user|text|intent)', input_string) if part.strip()]

    # Initialize the dictionary
    result_dict = {}

    # Current key placeholder
    current_key = None

    # Iterate over each part
    for part in parts:
        if part in ['user', 'text', 'intent']:
            current_key = part  # Update the current key
        else:
            # Remove leading/trailing spaces, colons, and quotes
            previous = None
            current = part.strip()
            while current != previous:
                previous = current
                current = current.strip(" ,:;'\"").strip()
            cleaned_value = current
            if current_key:
                result_dict[current_key] = cleaned_value

    return result_dict

def paraphrase_row(row):
    u1 = row['utterance1']
    u2 = row['utterance2']
    
    u1 = parse_key_value_pairs(u1)
    u2 = parse_key_value_pairs(u2)

    print(f"Original utterance1: {u1}")
    print(f"Original utterance2: {u2}")
    
    
    u1['text'] = paraphrase_text(question = u1['text'], num_return_sequences=1)[0]
    u2['text'] = paraphrase_text(question = u2['text'], num_return_sequences=1)[0]

    row['utterance1'] = json.dumps(u1)
    row['utterance2'] = json.dumps(u2)
    
    print(f"Augmented utterance1: {u1}")
    print(f"Augmented utterance2: {u2}")
    return row

# Load the dataset
df = pd.read_csv('train.csv')

# Filter rows where label is 1
df_label1 = df[df['label'] == 1].copy()

# Apply paraphrasing to these rows
augmented_rows = df_label1.apply(paraphrase_row, axis=1)

# Append the augmented rows to the original dataframe
df_augmented = pd.concat([df, augmented_rows])

# Save the augmented dataset
df_augmented.to_csv('train_augmented.csv', index=False)
