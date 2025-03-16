from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_tokenizer_model():
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=5  # Update based on your label count
    )
    return tokenizer, model