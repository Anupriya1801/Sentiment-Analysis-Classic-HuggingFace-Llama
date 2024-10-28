from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
import joblib

# Load and preprocess dataset
df = pd.read_csv('dataset.csv')
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)  # Encode target

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_data = Dataset.from_pandas(train_df)
test_data = Dataset.from_pandas(test_df)

# Load pre-trained tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the data
def tokenize_data(batch):
    return tokenizer(batch['review'], padding=True, truncation=True)

train_data = train_data.map(tokenize_data, batched=True)
test_data = test_data.map(tokenize_data, batched=True)

# Set format for PyTorch
train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01
)

# Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./sentiment_transformer_model')
tokenizer.save_pretrained('./sentiment_transformer_model')
