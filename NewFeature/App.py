from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the tokenizer (make sure this is the same tokenizer you used during training)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load the model architecture with the correct number of labels (adjust num_labels if needed)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)

# Load the saved state dictionary (weights)
checkpoint_path = 'C:/Users/Adnan Fatawi/Documents/Python/NLPenv/checkpoints/model_only_state_dict.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda')))

# Set model to evaluation mode
model.eval()

# Function to predict emotion
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()

    class_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    return class_labels[predicted_class]

# Serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint to receive text and run the model
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data['input_text']

    # Call the function to predict the emotion
    prediction = predict_emotion(user_input)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
