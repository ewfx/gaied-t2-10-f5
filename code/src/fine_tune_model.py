from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import torch
import schedule
import time
import json
from datasets import Dataset
import os

# Load and split dataset
dataset = load_dataset('csv', data_files={'full': 'emails.csv'})
train_test = dataset['full'].train_test_split(test_size=0.2, shuffle=True)

# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_test['train']['label']),
    y=train_test['train']['label']
)
weights = torch.tensor(class_weights, dtype=torch.float)

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

train_dataset = train_test['train'].map(tokenize_function, batched=True)
valid_dataset = train_test['test'].map(tokenize_function, batched=True)

# Custom trainer with class weights
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir='./logs',
    logging_steps=10,
    remove_unused_columns=False
)

# Initialize trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics
)

def load_training_data_from_jsonl(jsonl_path="training_data.jsonl"):
    """
    Load training data from a JSONL file for periodic retraining.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Training data file not found: {jsonl_path}")
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    
    # Map string labels to numerical values
    unique_labels = list(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    numerical_labels = [label_mapping[label] for label in labels]
    
    dataset = Dataset.from_dict({"text": texts, "label": numerical_labels})
    return dataset.train_test_split(test_size=0.2, shuffle=True)

def periodic_training(jsonl_path="training_data.jsonl"):
    """
    Trigger the training module using the training data from the JSONL file.
    """
    print("Starting periodic training...")
    
    # Load training data
    train_test = load_training_data_from_jsonl(jsonl_path)
    
    # Tokenize datasets
    train_dataset = train_test['train'].map(tokenize_function, batched=True)
    valid_dataset = train_test['test'].map(tokenize_function, batched=True)
    
    # Update the trainer with new datasets
    trainer.train_dataset = train_dataset
    trainer.eval_dataset = valid_dataset
    
    # Start training
    trainer.train()
    print("Periodic training complete!")
    trainer.save_model("./fine-tuned-model")
    tokenizer.save_pretrained("./fine-tuned-model")
    print("Model saved to ./fine-tuned-model")

# Scheduler to trigger training periodically
def start_scheduler():
    """
    Start the scheduler to trigger periodic training.
    """
    schedule.every().day.at("02:34").do(periodic_training, jsonl_path="training_data.jsonl")  # Adjust time as needed
    
    print("Scheduler started. Waiting for the next training cycle...")
    while True:
        schedule.run_pending()
        time.sleep(1)



if __name__ == "__main__":
    # Train and save model
    print("Starting training...")
    trainer.train()
    print("Training complete!")
    trainer.save_model("./fine-tuned-model")
    tokenizer.save_pretrained("./fine-tuned-model")
    print("Model saved to ./fine-tuned-model")
    
    # Start the scheduler
    start_scheduler()