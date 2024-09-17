import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Load the trained model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7)
# Load the checkpoint
checkpoint = torch.load("C:/Users/Adnan Fatawi/Documents/Python/NLPenv/checkpoints/model_epoch_3.pt", map_location=device)

# Load only the model's state_dict (weights)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Emotion labels mapping (7 classes as per your training)
emotion_labels = ['anger','disgust','fear','joy','neutral','sadness','surprise']

# Function to preprocess and predict on an individual example
def predict_emotion(utterance):
    # Tokenize and encode the input
    inputs = tokenizer.encode_plus(
        utterance,
        add_special_tokens=True,  # Add [CLS] and [SEP] tokens
        max_length=128,  # Set max length according to the model's needs
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"  # Return as PyTorch tensors
    )
    
    # Move tensors to the GPU if available
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Perform inference (no gradient calculation needed)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Get the predicted emotion label
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return emotion_labels[predicted_class]

# Function to evaluate the model on a test dataset
def evaluate_model(dataloader):
    model.eval()  # Set model to evaluation mode

    predictions = []
    true_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1)

        predictions.extend(predicted_class.cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

    # Print a classification report
    print(classification_report(true_labels, predictions, target_names=emotion_labels))

# Test the model with individual examples
def main():
    # Individual example predictions
    examples = [
        "I am so happy today!",
        "My jaw dropped when I saw the gift",
        "I hate this.",
        "That scare me",
        "You know what! screw this, im out"
    ]

    for example in examples:
        predicted_emotion = predict_emotion(example)
        print(f"Utterance: {example}")
        print(f"Predicted Emotion: {predicted_emotion}\n")

    # Test dataset evaluation (assuming test_dataloader is defined elsewhere)
    # evaluate_model(test_dataloader)  # Uncomment this line if you want to run on test dataset

if __name__ == "__main__":
    main()
