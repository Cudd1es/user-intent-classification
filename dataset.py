import json
from collections import Counter
import matplotlib.pyplot as plt

# Load your dataset
with open("flat_intent_dataset.json", "r") as f:
    data = json.load(f)

# Count intents
intent_counts = Counter([item["intent"] for item in data])

# Print results
print("Intent Distribution:\n")
for intent, count in intent_counts.items():
    print(f"{intent}: {count}")

# Optional: Plot distribution
plt.figure(figsize=(10, 6))
plt.bar(intent_counts.keys(), intent_counts.values(), color='skyblue')
plt.title("Intent Distribution")
plt.xlabel("Intent")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()