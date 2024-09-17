import pandas as pd
import spacy #perlu mencari tahu kegunaan spacy di project ini
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load spaCy model for text processing and word vectoring
nlp = spacy.load('en_core_web_md')  # en_core_web_md includes word vectors

# Load the dataset
df = pd.read_csv("C:/Users/Adnan Fatawi/Documents/Python/HuggingFaceenv/Dataframe/emotion_classification.csv")

# Preprocess the utterances by removing unnecessary characters and tokenizing the text
def preprocess_text(text):
    doc = nlp(text)
    # You can add more preprocessing steps here if needed
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Apply the preprocessing to the 'utterance' column
df['processed_utterance'] = df['Utterance'].apply(preprocess_text)

# Generate word vectors for the 'processed_utterance' column
def get_word_vector(text):
    doc = nlp(text)
    return doc.vector  # Get the vector representation of the entire document

# Apply word vectoring
df['word_vector'] = df['processed_utterance'].apply(get_word_vector)

# Encode categorical variables (sentiment and emotion)
label_encoder_sentiment = LabelEncoder()
label_encoder_emotion = LabelEncoder()

df['sentiment_encoded'] = label_encoder_sentiment.fit_transform(df['Sentiment'])
df['emotion_encoded'] = label_encoder_emotion.fit_transform(df['Emotion'])

# Split the data into training, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save the preprocessed and split data to new CSV files
train_df.to_csv('train_dataset.csv', index=False)
val_df.to_csv('val_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

print("Preprocessing complete. The processed datasets have been saved as 'train_dataset.csv', 'val_dataset.csv', and 'test_dataset.csv'.")
