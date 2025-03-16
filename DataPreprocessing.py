import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_file="Independent_Medical_Reviews.csv"):
    try:
        nltk.download(['stopwords', 'punkt', 'wordnet'])
        
        # Load and verify data
        df = pd.read_csv(input_file)
        print("Original columns:", df.columns.tolist())
        
        # Column mapping based on your data structure
        df = df.rename(columns={
            "Findings": "text",
            "Diagnosis Category": "label"
        })
        
        # Validate required columns
        if not {'text', 'label'}.issubset(df.columns):
            missing = {'text', 'label'} - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Clean data
        df = df.dropna(subset=['text', 'label'])
        df['text'] = df['text'].str.strip()
        
        # Text preprocessing
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        def clean_text(text):
            text = text.lower()
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(word) for word in tokens 
                     if word.isalpha() and word not in stop_words]
            return " ".join(tokens)
        
        df['clean_text'] = df['text'].apply(clean_text)
        
        # Label encoding
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])
        
        # Save processed data
        df.to_csv("preprocessed_medical_data.csv", index=False)
        print("Preprocessing completed successfully!")
        return df
        
    except Exception as e:
        print(f"Preprocessing failed: {str(e)}")
        return None

if __name__ == "__main__":
    preprocess_data()