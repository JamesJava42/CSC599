from transformers import TrainingArguments

def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,            # Try [1e-5, 2e-5, 3e-5]
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,           # Try [3, 4, 5]
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
    )
