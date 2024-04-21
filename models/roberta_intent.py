import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel, AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default='train.csv')
parser.add_argument('--test_data', type=str, default='test.csv')
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

def split_utterances(df):
    splits1 = df['utterance1'].str.split("\'")
    df['user1'], df['text1'], df['intent1'] =  splits1.str[3], splits1.str[7], splits1.str[11]
    splits2 = df['utterance2'].str.split("\'")
    df['user2'], df['text2'], df['intent2'] =  splits2.str[3], splits2.str[7], splits2.str[11]
    print(f"Train DF sample: {train_df.iloc[0, 2:]}")

train_file_path = args.train_data
test_file_path = args.test_data
print(f"Train dataset: {train_file_path}, Test dataset: {test_file_path}")
train_df = pd.read_csv(train_file_path)
split_utterances(train_df)
# print(f"Train DF sample: {train_df.iloc[0, :]}")
test_df = pd.read_csv(test_file_path)
split_utterances(test_df)

all_intents = set(train_df['intent1']) | set(test_df['intent1']) | set(train_df['intent2']) | set(test_df['intent2'])
intent_to_index = {intent: idx for idx, intent in enumerate(sorted(all_intents))}
user_to_index = {"Alien": 0, "Human": 1}

class RobertaWithCategory(torch.nn.Module):
    def __init__(self, bert_model, category_size):
        super().__init__()
        self.bert = bert_model
        self.category_size = category_size
        # Assuming binary classification. Change the output size as needed.
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size + category_size, 2)

    def forward(self, input_ids, attention_mask, category1, category2):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the pooled output
        pooled_output = outputs[1]
        
        # Concatenate pooled output with category vector
        combined = torch.cat((pooled_output, category1, category2), dim=1)
        
        # Pass the combined tensor through the classifier to get logits
        logits = self.classifier(combined)
        return logits


class DialogueDataset(Dataset):
    def __init__(self, utterances, labels, tokenizer, intents1, intents2):
        self.utterances = utterances
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = 149

        # Create a mapping from category names to indices
        unique_intents = sorted(set(intents1) | set(intents2))
        self.intents1 = intents1
        self.intents2 = intents2
        self.intent_to_index = {intent: idx for idx, intent in enumerate(unique_intents)}
        self.num_categories = len(unique_intents)
    
    def one_hot_encode_intent(self, intent):
        one_hot = [0] * len(all_intents)
        intent_index = intent_to_index[intent]
        one_hot[intent_index] = 1
        return one_hot

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

        intent1 = self.one_hot_encode_intent(self.intents1[item])
        intent2 = self.one_hot_encode_intent(self.intents2[item])

        return {
            'utterance_text': utterance,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'intent1': torch.tensor(intent1, dtype=torch.float),
            'intent2': torch.tensor(intent2, dtype=torch.float)
        }

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
MAX_LEN = 128  

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = DialogueDataset(
        # utterances="[CLS] " + df['text1'] + " [" + df['user1'] + "]" + " [" + df['intent1'] + "]" + " [SEP] " +  
        # df['text2'] + " [" + df['user2'] + "]" + " [" + df['intent2'] + "]",
        utterances="[CLS] " + df['user1'] + " : " +  df['utterance1'] + " [SEP] "  + df['user1'] + " : " + df['utterance2'],  # Concatenate utterances
        labels=df['label'],
        intents1=df['intent1'],
        intents2=df['intent2'],
        tokenizer=tokenizer
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0)

BATCH_SIZE = 256
EPOCHS = args.epochs

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

num_categories = len(all_intents)*2

model = RobertaModel.from_pretrained('roberta-base', 
        num_labels=2,         
        attention_probs_dropout_prob=0.2,
        hidden_dropout_prob=0.2
    )

device = "cuda:4" if torch.cuda.is_available() else "cpu"
# model = model.to(device)
model = RobertaWithCategory(model, num_categories).to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
print(f"Optimizer params: {optimizer}")
# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

def train_epoch(model, data_loader, optimizer, device, n_examples):
    model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        intent1 = d['intent1'].to(device)
        intent2 = d['intent2'].to(device)
        labels = d['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            category1=intent1,
            category2=intent2
        )

        # Compute the loss
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())

        # Get the predictions
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

        # Backward pass and optimize
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
            intent1 = d['intent1'].to(device)
            intent2 = d['intent2'].to(device)
            labels = d['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                category1=intent1,
                category2=intent2
            )

            # Compute the loss
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())

            # Get the predictions
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.float() / n_examples, np.mean(losses)

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

def get_predictions(model, data_loader, device):
    model = model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            inputs = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)
            intent1 = d['intent1'].to(device)
            intent2 = d['intent2'].to(device)

            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                category1=intent1,
                category2=intent2
            )

            _, preds = torch.max(outputs, dim=1)

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
report = classification_report(y_test, y_pred, target_names=['Segment Continuation', 'Segment Shift'], output_dict=True)
import json
with open('segment_break_mlm_gpt.json', 'w') as f:
    json.dump(report, f, indent=4)