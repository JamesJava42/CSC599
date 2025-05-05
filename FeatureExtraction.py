from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd

class FeatureExtractor:
    def __init__(self, max_features=500):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
    
    def fit_transform(self, texts: pd.Series):
        print("\n[2/5] Training TF-IDF vectorizer...")
        tfidf_matrix = self.vectorizer.fit_transform(texts.astype(str))
        self.plot_feature_distribution(tfidf_matrix)
        return tfidf_matrix
    
    def transform(self, texts: pd.Series):
        return self.vectorizer.transform(texts.astype(str))
    
    def plot_feature_distribution(self, tfidf_matrix):
        plt.figure(figsize=(12, 6))
        features = self.vectorizer.get_feature_names_out()[:10]
        importance = tfidf_matrix.sum(axis=0).A1[:10]
        
        plt.bar(features, importance)
        plt.title("Top 10 Important Clinical Terms (TF-IDF)")
        plt.xlabel("Medical Terms")
        plt.ylabel("Importance Score")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("clinical_terms_distribution.png")
        print("Saved feature importance plot: clinical_terms_distribution.png")