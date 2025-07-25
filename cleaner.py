import json

with open("flat_intent_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# remove duplicated samples
# use content + intent pair to check duplicates
unique_data = []
seen = set()
for item in data:
    key = (item["content"].strip(), item["intent"])
    if key not in seen:
        seen.add(key)
        unique_data.append(item)

# remove empty text
cleaned_data = [item for item in unique_data if item["content"].strip() != ""]

# remove sentences that are too long or too short
filtered_data = [item for item in cleaned_data if 5 <= len(item["content"]) <= 150]

# print the distribution of samples in different categories
from collections import Counter
intent_counts = Counter([item["intent"] for item in filtered_data])
print("distribution of intent counts:")
for intent, count in intent_counts.items():
    print(f"{intent}: {count}")

# save the cleaned data
with open("cleaned_intent_dataset.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"count of sample before cleaning：{len(data)}")
print(f"count of sample after cleaning：{len(filtered_data)}")