# Week 12: Sequence Models (NLP Intro) — Tutorial

## Why this week matters
Text is sequential and variable-length — very different from images or
tabular data. This week covers how raw text becomes numbers a model can
learn from, and the architectures built to handle sequences.

---

## 1. Text Preprocessing

```python
text = "The movie was surprisingly good!"

# Tokenization: split into units the model will process
tokens = text.lower().split()   # simplest version: ["the", "movie", ...]

# In practice, use a proper tokenizer (handles punctuation, subwords, etc.)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(["The movie was good", "The movie was bad"])
```

**Embeddings** (Word2Vec/GloVe) map each word to a dense vector such that
similar words end up close together in vector space — e.g., "king" and
"queen" are nearer each other than "king" and "banana."

```python
# Conceptual: word2vec turns "good" -> [0.12, -0.4, 0.88, ...] (e.g. 100-dim)
```

---

## 2. RNNs and LSTMs (Conceptually)

A plain neural net has no memory between inputs. An **RNN** processes a
sequence one token at a time, carrying a "hidden state" forward — giving it
memory of what came before.

```python
import torch.nn as nn
rnn = nn.LSTM(input_size=100, hidden_size=128, batch_first=True)
```

LSTMs (Long Short-Term Memory) add gates that control what to remember,
forget, and output — this fixes plain RNNs' tendency to "forget" long-range
context (the vanishing gradient problem).

---

## 3. Attention and Transformers (Conceptual Preview)

RNNs process sequentially, which is slow and struggles with very long-range
dependencies. **Attention** lets a model directly weigh how relevant every
other token is to the current one, regardless of distance — this is the
core idea behind Transformers, which you'll go deeper on next week.

---

## 4. A Simple Sentiment Classifier

```python
class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)   # positive / negative

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])
```

---

## Project
Build a sentiment analysis model (movie reviews or tweets):
- [ ] Preprocess text: tokenize, build a vocabulary, pad sequences
- [ ] Train an embedding + LSTM classifier
- [ ] Report accuracy on a held-out test set

## Resources
- Jay Alammar: Illustrated Word2Vec — https://jalammar.github.io/illustrated-word2vec/
- StatQuest: RNNs and LSTMs — https://www.youtube.com/watch?v=YCzL96nL7j0
- Dataset: IMDB reviews — https://ai.stanford.edu/~amaas/data/sentiment/
