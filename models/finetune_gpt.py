from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, pipeline
import torch
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def prepare_data(file_path, tokenizer):
    data = pd.read_csv(file_path)
    data['inputs'] = "Utterance1: " + data['utterance1'].astype(str) + " Utterance2: " + data['utterance2'].astype(str)
    data['labels'] = data['label'].map(lambda x: 'Yes' if x == 1 else 'No')
    return data

def tokenize_function(examples):
    model_inputs = tokenizer(examples['inputs'], padding="max_length", truncation=True, max_length=128)
    labels = []
    for label in examples['labels']:
        label_ids = tokenizer.encode(label, add_special_tokens=False)
        label_ids_padded = label_ids + [-100] * (128 - len(label_ids))
        labels.append(label_ids_padded)
    model_inputs['labels'] = labels
    return model_inputs

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model = model.to('cuda')  # Ensure model is on GPU
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_data = prepare_data('../train.csv', tokenizer)
test_data = prepare_data('../test.csv', tokenizer)
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_test_datasets = test_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./generated_results',
    num_train_epochs=50,
    per_device_train_batch_size=50,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_test_datasets
)
trainer.train()

# Make sure to explicitly move tensors to the correct device when using the generator
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

test_data['predicted'] = [generator(f"Utterance1: {item['utterance1']} Utterance2: {item['utterance2']}", max_new_tokens=50)[0]['generated_text'] for item in test_data.to_dict('records')]

def extract_yes_no(prediction):
    return 'Yes' if 'Yes' in prediction else 'No'

test_data['predicted_label'] = test_data['predicted'].apply(extract_yes_no)
true_labels = test_data['labels'].map(lambda x: 1 if x == 'Yes' else 0)
predicted_labels = test_data['predicted_label'].map(lambda x: 1 if x == 'Yes' else 0)

accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {fscore:.4f}")

try :
    from sklearn.metrics import classification_report
    report = classification_report(true_labels, predicted_labels, target_names=['Segment Continuation', 'Segment Shift'], output_dict=True)
    import json
    with open(f'../results/hello.json', 'w') as f:
        json.dump(report, f, indent=4)
except:
    pass
