import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load datasets
train_df = pd.read_csv('train_dataset.csv')
val_df = pd.read_csv('val_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')

# Check for any NaN values in the processed_utterance column
train_df['processed_utterance'].fillna('', inplace=True)
val_df['processed_utterance'].fillna('', inplace=True)
test_df['processed_utterance'].fillna('', inplace=True)

# Ensure all processed_utterance values are strings
train_df['processed_utterance'] = train_df['processed_utterance'].astype(str)
val_df['processed_utterance'] = val_df['processed_utterance'].astype(str)
test_df['processed_utterance'] = test_df['processed_utterance'].astype(str)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Ensure GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Tokenization function with debugging
def tokenize_function(example):
    print(f"Tokenizing: {example['processed_utterance']}")
    return tokenizer(
        example['processed_utterance'], 
        padding='max_length', 
        truncation=True, 
        max_length=512
    )

# Convert to Hugging Face Dataset format and apply tokenization
train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

# Ensure the labels are properly mapped
train_dataset = train_dataset.map(lambda examples: {'labels': examples['sentiment_encoded']})
val_dataset = val_dataset.map(lambda examples: {'labels': examples['sentiment_encoded']})
test_dataset = test_dataset.map(lambda examples: {'labels': examples['sentiment_encoded']})

# Format dataset for PyTorch with labels included
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the Trainer with train and validation datasets including labels
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate(eval_dataset=test_dataset)
print(eval_result)

