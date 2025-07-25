# predict_intent_cli.py

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json

# Detect device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = DistilBertTokenizer.from_pretrained("./saved_model")
model.to(device)
model.eval()

# Load label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

# Prediction function
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label_id = torch.argmax(outputs.logits, dim=1).item()
    return id2label[predicted_label_id]

# CLI loop
print("Intent Classifier is ready. Type your sentence or 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break
    predicted = predict_intent(user_input)
    print(f"Predicted intent: {predicted}")