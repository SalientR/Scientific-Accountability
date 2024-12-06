import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import (BertTokenizer, BertForSequenceClassification, AdamW,
                          RobertaTokenizer, RobertaForSequenceClassification,
                          DebertaV2Tokenizer, DebertaForSequenceClassification)
import torch

# Load the CSV file
df = pd.read_csv('Classification_Ranking.csv', sep=';')

# Preprocess: Map labels to binary values
df['Label'] = df['Label'].map({'R': 1, 'NR': 0})

# Prepare text data by combining relevant text columns
text_columns = ['abstract', 'Materials_and_Methods', 'Conclusion', 'Results']
df[text_columns] = df[text_columns].fillna('')  # Fill NA with empty strings
text_features = df[text_columns].agg(' '.join, axis=1)

# Split the data into training and testing sets
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(text_features, df['Label'], test_size=0.2, random_state=42)

# Define custom Dataset class for BERT
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Function to train the model
def train_model(train_texts, train_labels, val_texts, val_labels, model_info, epochs=1, batch_size=8):
    model_class, tokenizer_class, model_name = model_info
    tokenizer = tokenizer_class.from_pretrained(model_name)
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=128)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model_class.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs} completed.')

    # Evaluate the model
    model.eval()
    val_predictions = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            val_predictions.extend(predictions)

    return val_predictions

# Define models and their associated tokenizers
model_names = {
    "DeBERTa": (DebertaForSequenceClassification, DebertaV2Tokenizer, "microsoft/deberta-v2-xxlarge"),
    "BERT-base": (BertForSequenceClassification, BertTokenizer, "bert-base-uncased"),
    "SciBERT": (BertForSequenceClassification, BertTokenizer, "allenai/scibert_scivocab_uncased"),
    "BioBERT": (BertForSequenceClassification, BertTokenizer, "michiyasunaga/BioBERT-base"),
    "RoBERTa": (RobertaForSequenceClassification, RobertaTokenizer, "roberta-base")
}

# Train each model and evaluate performance
for model_name, model_info in model_names.items():
    print(f"\nTraining: {model_name}")
    predictions = train_model(X_train_text.values, y_train_text.values, X_test_text.values, y_test_text.values, model_info, epochs=3, batch_size=8)

    # Evaluate model performance
    accuracy = accuracy_score(y_test_text, predictions)
    f1 = f1_score(y_test_text, predictions)
    print(f"{model_name} Model Accuracy: {accuracy:.4f}")
    print(f"{model_name} Model F1 Score: {f1:.4f}")
    print(f"{model_name} Model Classification Report:\n{classification_report(y_test_text, predictions)}")
    print(f"{model_name} Model Confusion Matrix:\n{confusion_matrix(y_test_text, predictions)}")
