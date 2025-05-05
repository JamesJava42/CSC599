import pandas as pd
import re
import nltk
import os
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class DataPreprocessor:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        words = [self.stemmer.stem(word) for word in words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    
    def load_and_preprocess(self, data_path: str) -> pd.DataFrame:
        print(f"\n[1/5] Loading dataset from {os.path.abspath(data_path)}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {os.path.abspath(data_path)}")
            
        df = pd.read_csv(data_path).head(20)
        
        # Validate required columns
        required_columns = ['Findings', 'Determination']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing columns: {missing_cols}")
            
        # Clean and preprocess
        df['processed_text'] = df['Findings'].apply(self.clean_text).replace('', np.nan)
        df = df.dropna(subset=['processed_text'])
        
        print("\nSample preprocessed data:")
        print(df[['Findings', 'processed_text', 'Determination']].head(2))
        return df[['processed_text', 'Determination']]