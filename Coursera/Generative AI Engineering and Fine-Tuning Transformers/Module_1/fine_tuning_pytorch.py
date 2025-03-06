from transformers import TrainingArguments
from transformers import AutoConfig, AutoModelForSequenceClassification
from torch.optim.adamw import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
from torch.optim import SGD

from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


# noinspection PyUnresolvedReferences
# Step 1: Load dataset
raw_datasets = load_dataset("glue", "mrpc")
raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select(range(int(0.05 * len(raw_datasets["train"]))))
raw_datasets["validation"] = raw_datasets["validation"].shuffle(seed=42).select(range(int(0.05 * len(raw_datasets["validation"]))))
raw_datasets["test"] = raw_datasets["test"].shuffle(seed=42).select(range(int(0.05 * len(raw_datasets["test"]))))

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config = AutoConfig.from_pretrained(checkpoint, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)
model.to(device)

# optimizer = SGD(model.parameters(), lr=5e-3, momentum=0.9)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
