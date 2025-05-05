from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class PerformanceEvaluator:
    metrics = []
    
    @staticmethod
    def evaluate_model(y_true, y_pred, model_name='Model'):
        present_labels = np.unique(np.concatenate([y_true, y_pred]))
        str_labels = [str(label) for label in present_labels]
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            labels=present_labels,
            target_names=str_labels,
            output_dict=True,
            zero_division=0
        )
        
        # Store metrics
        PerformanceEvaluator.metrics.append({
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred, average='weighted'),
            'Precision': report['weighted avg']['precision']
        })
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=present_labels)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=str_labels,
                    yticklabels=str_labels)
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"{model_name}_confusion_matrix.png")
        print(f"Saved {model_name} evaluation visualizations")

    @staticmethod
    def show_comparison():
        df = pd.DataFrame(PerformanceEvaluator.metrics)
        print("\n=== Model Performance Comparison ===")
        print(df)
        
        plt.figure(figsize=(12, 6))
        df.set_index('Model').plot(kind='bar', rot=45)
        plt.title("Model Performance Comparison")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig("model_comparison.png")
        print("Saved model comparison chart")