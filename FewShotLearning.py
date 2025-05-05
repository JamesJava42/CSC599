from transformers import pipeline

class FewShotClassifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def predict(self, texts, candidate_labels):
        predictions = []
        for text in texts:
            result = self.classifier(text, candidate_labels)
            predictions.append(result['labels'][0])
        return predictions