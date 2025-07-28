import json
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# handle the issue of mismatch between the model and the input tensor devices when using Apple’s MPS backend.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# load data
with open("cleaned_intent_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["content"] for item in data]
labels = [item["intent"] for item in data]

# map label to id
"""
"0": "benefits_value_alignment", 
"1": "commitment_decision_cues", 
"2": "cost_fee_inquiry", 
"3": "greeting_rapport_building", 
"4": "negotiation_signals", 
"5": "next_steps_logistics", 
"6": "objections_concerns", 
"7": "personal_goals_articulation", 
"8": "product_discovery", 
"9": "risk_suitability_assessment"
"""

label2id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
id2label = {idx: label for label, idx in label2id.items()}


with open("label_map.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)

y = [label2id[label] for label in labels]

# divide training and validation set
X_train, X_val, y_train, y_val = train_test_split(texts, y, test_size=0.2, random_state=42)

# use bert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# build Dataset format
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_dict({"text": X_train, "label": y_train}).map(tokenize_function, batched=True)
val_dataset = Dataset.from_dict({"text": X_val, "label": y_val}).map(tokenize_function, batched=True)

# load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label2id))
model.to(device)

# set the training args
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

# build Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# training
trainer.train()
trainer.evaluate()


model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

print("Training complete. Model and tokenizer saved to ./saved_model")