from transformers import AutoTokenizer

class LLMTokenizer:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        print(f"\nInitializing tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer vocab size:", self.tokenizer.vocab_size)
    
    def tokenize_batch(self, texts: list, max_length=128) -> dict:
        print("\nTokenizing sample text:", texts[0][:50] + "...")
        result = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        print("Tokenized output structure:", {k: v.shape for k, v in result.items()})
        return result