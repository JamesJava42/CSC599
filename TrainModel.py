from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import joblib

class ModelTrainer:
    def __init__(self):
        self.training_history = []
    
    def train_svm(self, X_train, y_train, C=1.0, kernel='linear'):
        print("\n[3/5] Training SVM classifier...")
        start_time = time.time()
        
        classes, counts = np.unique(y_train, return_counts=True)
        weights = class_weight.compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train
        )
        class_weights = dict(zip(classes, weights))
        
        model = SVC(
            C=C,
            kernel=kernel,
            class_weight=class_weights,
            random_state=42,
            probability=True,
            decision_function_shape='ovr'
        )
        model.fit(X_train, y_train)
        
        print(f"Training time: {time.time()-start_time:.1f}s")
        return model

    def train_naive_bayes(self, X_train, y_train):
        print("\n[3/5] Training Naive Bayes classifier...")
        start_time = time.time()
        model = MultinomialNB()
        model.fit(X_train, y_train)
        print(f"Training time: {time.time()-start_time:.1f}s")
        return model

    def train_random_forest(self, X_train, y_train):
        print("\n[3/5] Training Random Forest classifier...")
        start_time = time.time()
        model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_estimators=100
        )
        model.fit(X_train, y_train)
        print(f"Training time: {time.time()-start_time:.1f}s")
        return model

    def train_bert(self, dataset, model_name="emilyalsentzer/Bio_ClinicalBERT", num_labels=2):
        print(f"\n[4/5] Initializing {model_name} model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            evaluation_strategy="epoch",
            logging_steps=10,
            learning_rate=2e-5,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test']
        )
        
        print("Beginning fine-tuning...")
        training_result = trainer.train()
        self.plot_training_curve(training_result)
        return trainer

    def plot_training_curve(self, training_result):
        plt.figure(figsize=(10, 6))
        losses = [log['loss'] for log in training_result.state.log_history if 'loss' in log]
        plt.plot(losses, label='Training Loss')
        plt.title("LLM Training Progress")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_curve.png")
        print("Saved training curve visualization: training_curve.png")

    def save_model(self, model, model_type='svm', path='./models'):
        import os
        os.makedirs(path, exist_ok=True)
        
        if model_type == 'svm':
            joblib.dump(model, f"{path}/svm_model.pkl")
        elif model_type == 'bert':
            model.save_pretrained(f"{path}/bert_model")
        print(f"Saved {model_type.upper()} model to {path}")