import numpy as np
from transformers import Trainer , AutoProcessor
from transformers import TrainingArguments
from transformers import AutoConfig, AutoModelForSequenceClassification
from torch.optim.adamw import AdamW
from transformers import get_scheduler
import torch
import evaluate
from torch.optim import SGD

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


# Step 1: Load dataset
raw_datasets = load_dataset("glue", "mrpc")
raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select(range(int(0.05 * len(raw_datasets["train"]))))
raw_datasets["validation"] = raw_datasets["validation"].shuffle(seed=42).select(range(int(0.05 * len(raw_datasets["validation"]))))
raw_datasets["test"] = raw_datasets["test"].shuffle(seed=42).select(range(int(0.05 * len(raw_datasets["test"]))))

# Step 2: Tokenize the data
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Step 3: Define the data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Step 4: Define the training arguments
training_args = TrainingArguments(
    output_dir="fine_tuned_model",
    eval_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,   # Optional: applies L2 regularization
    warmup_ratio=0,  # Keep warmup steps at 0
    logging_steps=10,  # Logging frequency
)

processor = AutoProcessor.from_pretrained("bert-base-uncased")

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

config = AutoConfig.from_pretrained(checkpoint, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)
model.to(device)

num_training_steps = (len(tokenized_datasets["train"]) // training_args.per_device_train_batch_size) * training_args.num_train_epochs
# optimizer = SGD(model.parameters(), lr=5e-3, momentum=0.9)
optimizer = AdamW(model.parameters(), lr=5e-5)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,

)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=processor,  # Use processing_class instead of tokenizer
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler)  # Pass manually defined optimizer & scheduler

)

# start fine-tuning
trainer.train()
# Evaluate the model and save it
trainer.evaluate()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
