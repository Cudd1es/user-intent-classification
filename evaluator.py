import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import Dataset
import torch
import numpy as np

# Load label maps
with open("label_map.json", "r") as f:
    label_map = json.load(f)
    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

# Load saved model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("./saved_model")
model = DistilBertForSequenceClassification.from_pretrained("./saved_model")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

# Load your evaluation/test dataset (same format as before)
with open("cleaned_intent_dataset.json", "r") as f:
    data = json.load(f)

texts = [item["content"] for item in data]
true_labels = [label2id[item["intent"]] for item in data]

# Tokenize inputs
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
inputs = {key: val.to(device) for key, val in inputs.items()}

# Run model
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, axis=1).cpu().numpy()

# Convert to numpy array
true_labels = np.array(true_labels)

# Evaluation metrics
accuracy = accuracy_score(true_labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="weighted")
conf_matrix = confusion_matrix(true_labels, preds)
class_report = classification_report(true_labels, preds, target_names=[id2label[i] for i in range(len(id2label))])

# Print results
print("=== Evaluation Results ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)