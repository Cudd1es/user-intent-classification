# Intent Classification with DistilBERT

This project implements a complete intent classification pipeline for English-language user queries, using `DistilBERT`. It is designed for financial product sales assistants that require classifying user intent from short text inputs.

---

## Project Structure

```
├── cleaned_intent_dataset.json       # Cleaned and labeled intent dataset
├── label_map.json                    # Label-to-ID mapping
├── saved_model/                      # Trained model and tokenizer directory
├── cleaner.py                        # (Optional) Script to clean raw data
├── bert_training.py                  # Train DistilBERT on labeled dataset
├── evaluator.py                      # Evaluate model performance
├── intent_classifier.py              # Load model and classify CLI user inputs
└── README.md                         # This file
```

---

## Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
```


>  For Apple Silicon/Mac M1/M2: make sure your PyTorch installation supports `mps` if you want hardware acceleration.

---

## Step 1: Prepare Dataset

If you haven't done it yet, prepare your dataset in the format:

```json
[
  {
    "content": "What products do you offer?",
    "intent": "product_info"
  },
  ...
]
```

Place it in `cleaned_intent_dataset.json`.

Optionally, use `cleaner.py` to process raw data into this format.

---

##  Step 2: Train the Model

Run:

```bash
python bert_training.py
```

This will:
- Load and tokenize your dataset
- Train a DistilBERT classifier
- Save the trained model and tokenizer to `./saved_model/`
- Save the label mapping as `label_map.json`

---

##  Step 3: Evaluate Model

Run:

```bash
python evaluator.py
```

This will:
- Load the saved model and tokenizer
- Evaluate it on the dataset or your own test set
- Output metrics: **Accuracy, Precision, Recall, F1**, and **Confusion Matrix**

---

##  Step 4: Use CLI Intent Classifier

You can classify user input interactively:

```bash
python intent_classifier.py
```

Then input a query like:

```
> Let me check your pricing plans
Predicted Intent: product_info
```

---

##  Notes

- Model is based on `distilbert-base-uncased`
- Tokenizer max length is set to 128
- If your dataset grows, you may want to increase `num_train_epochs` or batch size in `bert_training.py`


---

##  License

MIT License or your custom license here