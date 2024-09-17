import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Load dataset
def load_data(file_path):
    # """ Load dataset from CSV file """
    return pd.read_csv(file_path)

# Preprocess text (tokenization and word vectoring)
def preprocess_text(text):
    # """ Tokenizes and vectorizes text using spaCy """
    doc = nlp(text)
    return np.mean([token.vector for token in doc if token.has_vector], axis=0)

# Encode emotion labels
def encode_labels(df, label_column):
    # """ Encodes labels into numerical format """
    encoder = LabelEncoder()
    df[label_column] = encoder.fit_transform(df[label_column])
    return df, encoder

# Apply TF-IDF for feature extraction
def apply_tfidf(df, text_column):
    # """ Applies TF-IDF to the utterances """
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df[text_column])
    return X_tfidf, vectorizer

# Split the dataset
def split_data(df, test_size=0.15, val_size=0.15):
    # """ Splits dataset into train, validation, and test sets """
    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data, val_data = train_test_split(train_data, test_size=val_size/(1 - test_size))
    return train_data, val_data, test_data

# Save datasets to CSV
def save_datasets(train_data, val_data, test_data, output_dir):
    # """ Saves the train, validation, and test sets to CSV files """
    train_data.to_csv(f'{output_dir}/train_data.csv', index=False)
    val_data.to_csv(f'{output_dir}/val_data.csv', index=False)
    test_data.to_csv(f'{output_dir}/test_data.csv', index=False)
    print(f"Datasets saved to {output_dir}")

# Main preprocessing function
def preprocess_data(file_path, output_dir):
    # """ Main function to handle full preprocessing """
    df = load_data(file_path)
    
    # Encode emotion labels
    df, label_encoder = encode_labels(df, 'Emotion')

    # Apply TF-IDF to the text column
    X_tfidf, vectorizer = apply_tfidf(df, 'Utterance')

    # Split the data
    train_data, val_data, test_data = split_data(df)

    # Save the datasets
    save_datasets(train_data, val_data, test_data, output_dir)

    # Return preprocessed data and encoders
    return train_data, val_data, test_data, label_encoder, vectorizer

# Example usage
if __name__ == '__main__':
    file_path = "C:/Users/Adnan Fatawi/Documents/Python/HuggingFaceenv/Dataframe/emotion_classification.csv"  # Replace with your actual file path
    output_dir = "C:/Users/Adnan Fatawi/Documents/Python/NLPenv/Dataframe"  # Replace with your desired output directory
    train_data, val_data, test_data, label_encoder, vectorizer = preprocess_data(file_path, output_dir)
    print("Preprocessing and saving complete!")
