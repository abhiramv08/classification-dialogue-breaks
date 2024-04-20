import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from transformers import BertModel

train_file_path = 'train.csv'  # Update this path
test_file_path = 'test.csv'    # Update this path
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

all_categories = set(train_df['category']) | set(test_df['category'])

# Step 2: Create a consistent category mapping
category_to_index = {category: idx for idx, category in enumerate(sorted(all_categories))}

class BertWithCategory(torch.nn.Module):
    def __init__(self, bert_model, category_size):
        super().__init__()
        self.bert = bert_model
        self.category_size = category_size
        # Assuming binary classification. Change the output size as needed.
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size + len(all_categories), 2)

    def forward(self, input_ids, attention_mask, category):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the pooled output
        pooled_output = outputs.pooler_output
        
        # Concatenate pooled output with category vector
        combined = torch.cat((pooled_output, category), dim=1)
        
        # Pass the combined tensor through the classifier to get logits
        logits = self.classifier(combined)
        return logits


# Preprocess the dataset
class DialogueDataset(Dataset):
    def __init__(self, utterances, labels, tokenizer, max_len,categories):
        self.utterances = utterances
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = 149

        # Create a mapping from category names to indices
        unique_categories = sorted(set(categories))
        self.categories = categories
        self.category_to_index = {category: idx for idx, category in enumerate(unique_categories)}
        

        self.num_categories = len(unique_categories)

    def one_hot_encode_category(self, category):
        one_hot = [0] * len(all_categories)
        category_index = category_to_index[category]
        one_hot[category_index] = 1
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
        
        category = self.one_hot_encode_category(self.categories[item])
        return {
            'utterance_text': utterance,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'category': torch.tensor(category, dtype=torch.float)
        }
    

# Define the tokenizer and maximum sequence length
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 149  # Set this based on your dataset analysis

# Create data loaders
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = DialogueDataset(
        utterances=df['utterance1'] + " [SEP] " + df['utterance2'],  # Concatenate utterances
        labels=df['label'],
        categories=df['category'],  # Include the category column
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=0)


BATCH_SIZE = 256
EPOCHS = 13  # Define the number of epochs

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define optimizer and move model to device
device = torch.device('cuda')
model = model.to(device)
num_categories = len(set(train_df['category']))

# Create an instance of BertWithCategory
bert_model = BertModel.from_pretrained('bert-base-uncased')
model = BertWithCategory(bert_model, num_categories).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training and evaluation functions
def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        categories = d['category'].to(device)
        labels = d['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, category=categories)
        
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


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            categories = d['category'].to(device)
            labels = d['labels'].to(device)

            # Forward pass (without labels)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, category=categories)

            # Compute the loss
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())

            # Get the predictions
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.float() / n_examples, np.mean(losses)


# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Train and evaluate the model
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    # Training
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        len(train_df)
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')

    # Evaluation
    loss_fn = torch.nn.CrossEntropyLoss()

    # Inside your testing loop
    test_acc, test_loss = eval_model(
        model,
        test_data_loader,
        loss_fn,
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
report = classification_report(y_test, y_pred, target_names=['Segment Continuation', 'Segment Shift'], output_dict=True)
import json
with open('bert_base_category.json', 'w') as f:
    json.dump(report, f, indent=4)
