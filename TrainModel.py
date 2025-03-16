import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from Tokenzation import get_tokenizer_model  # Fixed import

class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def train_llm():
    try:
        df = pd.read_csv("preprocessed_medical_data.csv")
        tokenizer, model = get_tokenizer_model()
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['clean_text'].tolist(),
            df['label'].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )
        
        # Create datasets
        train_dataset = MedicalDataset(train_texts, train_labels, tokenizer)
        test_dataset = MedicalDataset(test_texts, test_labels, tokenizer)
        
        # Training configuration
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=4,  # Reduced for medical text length
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=50,
            fp16=True  # Enable mixed precision
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )
        
        trainer.train()
        return trainer
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return None