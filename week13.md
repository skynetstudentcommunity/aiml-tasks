# Week 13: Transformers & Modern NLP — Tutorial

## Why this week matters
Transformers replaced RNNs as the dominant architecture for text (and now
much more) because attention processes whole sequences in parallel and
handles long-range dependencies far better.

---

## 1. Attention, In Depth

For each token, attention computes three vectors: **Query**, **Key**, and
**Value**. A token's output is a weighted sum of all tokens' Values, where
the weights come from comparing its Query against every token's Key —
"how relevant is each other word to me right now?"

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V


Multi-head attention runs several of these in parallel, letting the model
attend to different kinds of relationships (syntax, coreference, etc.)
simultaneously.

---

## 2. Using HuggingFace Transformers

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("This tutorial is actually pretty clear!")
# [{'label': 'POSITIVE', 'score': 0.999...}]
```

`pipeline` wraps tokenizer + model + post-processing — great for quick
experiments before you build a custom fine-tuning setup.

---

## 3. Fine-Tuning a Pretrained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
)

trainer = Trainer(model=model, args=training_args,
                   train_dataset=tokenized_dataset["train"],
                   eval_dataset=tokenized_dataset["test"])
trainer.train()
```

Fine-tuning takes a model that already understands language broadly and
specializes it on your specific task/dataset — needs far less data than
training a transformer from scratch.

---

## Project
Fine-tune a pretrained transformer on a text classification task:
- [ ] Load and tokenize your dataset (reuse Week 12's sentiment data if you like)
- [ ] Fine-tune DistilBERT with HuggingFace `Trainer`
- [ ] Compare accuracy against your Week 12 LSTM model

## Resources
- Jay Alammar: The Illustrated Transformer — https://jalammar.github.io/illustrated-transformer/
- HuggingFace NLP Course (free) — https://huggingface.co/learn/nlp-course
- HuggingFace Transformers docs — https://huggingface.co/docs/transformers/index
