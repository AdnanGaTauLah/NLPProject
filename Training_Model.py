import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch

# Assuming you have a function that loads the already split data
def load_pre_split_data():
    # Load pre-split training and validation sets (adjust paths accordingly)
    train_df = pd.read_csv("C:/Users/Adnan Fatawi/Documents/Python/NLPenv/Dataframe/train_data.csv")
    val_df = pd.read_csv("C:/Users/Adnan Fatawi/Documents/Python/NLPenv/Dataframe/val_data.csv")
    
    # Extract utterances and labels
    train_texts = train_df['Utterance'].tolist()
    train_labels = train_df['Emotion'].tolist()
    
    val_texts = val_df['Utterance'].tolist()
    val_labels = val_df['Emotion'].tolist()

    # Encode the emotion labels into integers
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    val_labels = label_encoder.transform(val_labels)
    
    return train_texts, train_labels, val_texts, val_labels

# Tokenize the input texts
def tokenize_function(examples, tokenizer, max_length=256):
    return tokenizer(examples, padding="max_length", truncation=True, max_length=max_length)

# Create a Dataset for BERT
def create_dataset(texts, labels, tokenizer, max_length=256):
    tokenized_inputs = tokenize_function(texts, tokenizer, max_length=max_length)
    dataset = Dataset.from_dict({
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': labels
    })
    return dataset

# Train the model
def train(model, train_loader, optimizer, lr_scheduler, num_epochs, device):
    model.train()  # Sets the model to training mode
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to the GPU (if available)
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            epoch_loss += loss.item()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(f"Training Loss: {epoch_loss / len(train_loader)}")
        
        # Save checkpoint after every epoch
        save_checkpoint(model, optimizer, epoch, epoch_loss / len(train_loader))

        
import os

def save_checkpoint(model, optimizer, epoch, loss, save_dir="checkpoints"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,}, model_save_path)

    print(f"Checkpoint saved at {model_save_path}")

        

# Helper function to convert the batch into the correct format
def collate_fn(batch):
    return {
        'input_ids': torch.tensor([item['input_ids'] for item in batch]),
        'attention_mask': torch.tensor([item['attention_mask'] for item in batch]),
        'labels': torch.tensor([item['labels'] for item in batch])
    }

# Evaluate the model
def evaluate(model, eval_loader, device):
    model.eval()
    total, correct = 0, 0
    for batch in tqdm(eval_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    # Set up device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load and split the dataset
    train_texts, train_labels, val_texts, val_labels = load_pre_split_data()
    
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(train_labels)))
    model.to(device)

    # Tokenize data
    train_dataset = create_dataset(train_texts, train_labels, tokenizer)
    val_dataset = create_dataset(val_texts, val_labels, tokenizer)

    # Create DataLoaders with collate_fn
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=25, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

    # Set the number of epochs
    num_epochs = 3
    
    # Set up optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=25e-6)
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Train the model
    train(model, train_loader, optimizer, lr_scheduler, num_epochs, device)

    # Evaluate the model
    print("\nEvaluating model:")
    evaluate(model, val_loader, device)

if __name__ == "__main__":
    main()