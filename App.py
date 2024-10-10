from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the tokenizer (ensure this is the same tokenizer used during training)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load the model architecture
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)

# Load the saved model state dictionary (weights)
checkpoint_path = 'C:/Users/Adnan Fatawi/Documents/Python/NLPenv/checkpoints/model_only_state_dict.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda')))

# Set model to evaluation mode
model.eval()

# Define the emotion classes
class_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Define a function to predict emotion based on input text
def predict_emotion(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Make predictions with the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply softmax to get probabilities and find the predicted class
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()
    
    # Return the predicted emotion label
    return class_labels[predicted_class]

# Define the main route for receiving requests and sending JSON responses
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    
    # Extract the user input from the JSON data
    user_input = data.get('input_text')
    
    # Check if input text was provided
    if not user_input:
        return jsonify({'error': 'No input text provided'}), 400
    
    # Predict the emotion
    prediction = predict_emotion(user_input)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
