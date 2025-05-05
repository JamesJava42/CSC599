from DataPreprocessing import DataPreprocessor
from FeatureExtraction import FeatureExtractor
from TrainModel import ModelTrainer
from PerformanceMatrix import PerformanceEvaluator
from Tokenization import LLMTokenizer
from FewShotLearning import FewShotClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from datasets import Dataset
import pandas as pd
import numpy as np

def main():
    try:
        # Initialize components
        preprocessor = DataPreprocessor()
        feature_extractor = FeatureExtractor(max_features=500)
        model_trainer = ModelTrainer()
        
        # Load and preprocess data
        data = preprocessor.load_and_preprocess("Independent_Medical_Reviews.csv")
        
        # Validate data
        if len(data) < 10:
            raise ValueError("Insufficient data samples (minimum 10 required)")
        
        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            data['processed_text'], 
            data['Determination'], 
            test_size=0.2, 
            stratify=data['Determination'],
            random_state=42
        )
        
        # Traditional Models Pipeline
        print("\n=== Traditional Machine Learning Models ===")
        X_tfidf_train = feature_extractor.fit_transform(X_train)
        X_tfidf_test = feature_extractor.transform(X_test)
        
        # Handle class imbalance
        print("\nOriginal class distribution:", y_train.value_counts())
        smote = SMOTE(sampling_strategy='auto', k_neighbors=1)
        X_res, y_res = smote.fit_resample(X_tfidf_train, y_train)
        print("Resampled class distribution:", pd.Series(y_res).value_counts())
        
        # Train and evaluate traditional models
        traditional_models = {
            'SVM': model_trainer.train_svm,
            'Naive Bayes': model_trainer.train_naive_bayes,
            'Random Forest': model_trainer.train_random_forest
        }
        
        for model_name, trainer in traditional_models.items():
            print(f"\n--- Training {model_name} ---")
            model = trainer(X_res, y_res)
            preds = model.predict(X_tfidf_test)
            PerformanceEvaluator.evaluate_model(y_test, preds, model_name)
            model_trainer.save_model(model, model_type=model_name.lower().replace(' ', '_'))
        
        # LLM Pipeline
        print("\n=== Large Language Models ===")
        tokenizer = LLMTokenizer()
        tokenized_train = tokenizer.tokenize_batch(X_train.tolist())
        tokenized_test = tokenizer.tokenize_batch(X_test.tolist())
        
        # Convert to Hugging Face dataset format
        bert_dataset = {
            'train': Dataset.from_dict(tokenized_train),
            'test': Dataset.from_dict(tokenized_test)
        }
        
        # Train and evaluate ClinicalBERT
        print("\n--- Training ClinicalBERT ---")
        bert_trainer = model_trainer.train_bert(bert_dataset)
        bert_outputs = bert_trainer.predict(bert_dataset['test'])
        bert_preds = np.argmax(bert_outputs.predictions, axis=1)
        PerformanceEvaluator.evaluate_model(y_test, bert_preds, 'ClinicalBERT')
        model_trainer.save_model(bert_trainer.model, model_type='bert')
        
        # Few-Shot Learning
        print("\n=== Few-Shot Learning ===")
        few_shot_clf = FewShotClassifier()
        candidate_labels = y_train.unique().tolist()
        few_shot_preds = few_shot_clf.predict(X_test.tolist(), candidate_labels)
        PerformanceEvaluator.evaluate_model(y_test, few_shot_preds, 'Few-Shot BART')
        
        # Show final comparison
        PerformanceEvaluator.show_comparison()
        
    except Exception as e:
        print(f"\nCritical Error: {str(e)}")

if __name__ == "__main__":
    main()