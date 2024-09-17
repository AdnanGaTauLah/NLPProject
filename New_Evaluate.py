import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# Load Dataset Function (you might already have this in your project)
def load_data(filename):
    # Assuming you have a method to load the test data as lists of texts and labels
    # Modify this function to match your data loading process
    # Here it's assumed you have CSV files for the dataset
    import pandas as pd
    df = pd.read_csv(filename)
    texts = df['Utterance'].tolist()
    labels = df['Emotion'].tolist()
    return texts, labels

# Create Dataset Function (same as used during training)
def create_dataset(texts, labels, tokenizer, max_len=128):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
    return dataset

# Evaluation Function
def evaluate(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            # Unpack the inputs from the dataloader and move them to the device
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            # Forward pass, get predictions
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits

            # Move logits to CPU and find predicted labels
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Compute accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1

# Function to Load the Model Checkpoint
def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")
    return model

def main():
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load the trained model checkpoint
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)  # Adjust num_labels as needed
    optimizer = None  # Not needed for evaluation, so can be ignored
    checkpoint_path = "C:/Users/Adnan Fatawi/Documents/Python/NLPenv/checkpoints/model_epoch_3.pt"  # Use your last saved checkpoint
    model = load_checkpoint(checkpoint_path, model, optimizer)
    model.to(device)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the test data
    test_texts, test_labels = load_data("C:/Users/Adnan Fatawi/Documents/Python/NLPenv/Dataframe/test_data.csv")  # Replace with your test dataset file
    test_dataset = create_dataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Evaluate the model on the test dataset
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
