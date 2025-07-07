# ticket_classifier_bert.py

import torch    #PyTorch, used for model training and inference.
import pandas as pd       # For handling tabular dataset.
from transformers import (   #transformers: Hugging Face library — 
                            # provides pre-trained models (like BERT), tokenizers, and 
                            #utilities for training (Trainer, TrainingArguments).
    BertTokenizer, 
    BertForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# 1. Sample Dataset 
data = {
    'text': [
        "My electricity bill is too high this month.",
        "There is a power outage in my area since morning.",
        "Meter reading is showing wrong value.",
        "Why is my power bill double this time?",
        "No electricity in sector 10 since yesterday.",
        "My meter is showing some problems."
    ],
    'label': ['Billing', 'Outage', 'Meter', 'Billing', 'Outage', 'Meter']
}
df = pd.DataFrame(data)       #  Converts them to a DataFrame using pandas.

# Label encoding          
label_map = {'Billing': 0, 'Outage': 1, 'Meter': 2}  #label_map assigns numeric IDs to classes for training (required for BERT).
reverse_label_map = {v: k for k, v in label_map.items()} #reverse_label_map is used later to convert predictions back to readable labels.
df['label'] = df['label'].map(label_map) #map the string labels to integers in the DataFrame.

