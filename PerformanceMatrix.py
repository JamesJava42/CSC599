import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_performance(trainer, test_dataset):
    try:
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        
        accuracy = accuracy_score(test_dataset.labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_dataset.labels, pred_labels, average='weighted'
        )
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return None