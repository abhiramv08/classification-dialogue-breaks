import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# Load the dataset
train_file_path = 'train.csv'  # Update this path
test_file_path = 'test.csv'    # Update this path
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Preprocess the dataset
class DialogueDataset(Dataset):
    def __init__(self, utterances, labels, tokenizer, max_len):
        self.utterances = utterances
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = 149

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, item):
        utterance = str(self.utterances[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            utterance,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'utterance_text': utterance,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Define the tokenizer and maximum sequence length
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128  # Set this based on your dataset analysis

# Create data loaders
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = DialogueDataset(
        utterances=df['utterance1'] + " [SEP] " + df['utterance2'],
        labels=df['label'],
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0)

BATCH_SIZE = 256
EPOCHS = 30  # Define the number of epochs

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define optimizer and move model to device
#device = torch.device('mps')
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training and evaluation functions
def train_epoch(model, data_loader, optimizer, device, n_examples):
    model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        losses.append(loss.item())

        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions.float() / n_examples, np.mean(losses)

def eval_model(model, data_loader, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            losses.append(loss.item())

            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.float() / n_examples, np.mean(losses)

# Train and evaluate the model
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        optimizer,
        device,
        len(train_df)
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')

    test_acc, test_loss = eval_model(
        model,
        test_data_loader,
        device,
        len(test_df)
    )
    print(f'Test loss {test_loss} accuracy {test_acc}')
    print()

# Generating a classification report
def get_predictions(model, data_loader, device):
    model = model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            inputs = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu())
            real_values.extend(labels.cpu())

    predictions = torch.stack(predictions).to(torch.float32).numpy()
    real_values = torch.stack(real_values).to(torch.float32).numpy()
    return predictions, real_values

y_pred, y_test = get_predictions(
    model,
    test_data_loader,
    device
)

print(classification_report(y_test, y_pred, target_names=['Segment Continuation', 'Segment Shift']))
