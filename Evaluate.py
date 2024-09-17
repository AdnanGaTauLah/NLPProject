import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load datasets
test_df = pd.read_csv('test_dataset.csv')

# Ensure all processed_utterance values are strings
test_df['processed_utterance'] = test_df['processed_utterance'].astype(str)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained("C:/Users/Adnan Fatawi/Documents/Python/NLPenv/checkpoints")  # Load the trained model

# Ensure GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Tokenization function
def tokenize_function(example):
    return tokenizer(
        example['processed_utterance'], 
        padding='max_length', 
        truncation=True, 
        max_length=512
    )

# Convert to Hugging Face Dataset format and apply tokenization
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

# Ensure the labels are properly mapped
test_dataset = test_dataset.map(lambda examples: {'labels': examples['sentiment_encoded']})

# Format dataset for PyTorch with labels included
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define a function to compute evaluation metrics
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Initialize the Trainer with the test dataset and the compute_metrics function
trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

# Evaluate the model on the test set
eval_result = trainer.evaluate(eval_dataset=test_dataset)

# Print evaluation results
print("Evaluation results on the test set:")
for key, value in eval_result.items():
    print(f"{key}: {value:.4f}")

# Optionally, test with an individual example
example_text = "This is a great day!"
inputs = tokenizer(example_text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
inputs = {key: value.to(device) for key, value in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
predicted_label = prediction.item()
print(f"Predicted label for example text: {predicted_label}")
