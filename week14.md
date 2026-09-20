# Week 14: Generative AI & LLMs — Tutorial

## Why this week matters
LLMs are the most visible application of everything you've learned so far.
This week is about using them effectively via API, not training one — that
takes resources far beyond a personal project.

---

## 1. How LLMs Work, at a High Level

An LLM is trained to predict the next token given everything before it. At
inference time, it generates text one token at a time, each new token
chosen based on the growing context (including its own previous outputs).

"The capital of France is" -> model predicts "Paris" with high probability


**Tokenization** breaks text into subword units (not always whole words) —
this is why LLMs sometimes struggle with things like counting letters in a
word: they see tokens, not characters.

---

## 2. Prompt Engineering Basics

Bad: "Summarize this."
Better: "Summarize the following article in 3 bullet points, focused on
the financial impact. Article: <text>"


Effective prompts are specific about: task, format, length, and any
constraints. Giving examples of desired input/output ("few-shot prompting")
often improves results further.

---

## 3. Calling an LLM API

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=500,
    messages=[{"role": "user", "content": "Summarize this in 3 bullets: ..."}]
)
print(response.content[0].text)
```

---

## 4. Intro to RAG (Retrieval-Augmented Generation)

LLMs don't know your private documents. RAG fixes this by: (1) splitting
your documents into chunks, (2) embedding each chunk into a vector, (3) at
query time, finding the most relevant chunks and feeding them into the
prompt alongside the question.

```python
# Conceptual flow:
chunks = split_into_chunks(document_text)
embeddings = [embed(chunk) for chunk in chunks]

query_embedding = embed(user_question)
top_chunks = find_most_similar(query_embedding, embeddings, chunks, k=3)

prompt = f"Answer using only this context:\n{top_chunks}\n\nQuestion: {user_question}"
response = call_llm(prompt)
```

This grounds the model's answer in your actual data rather than relying on
what it memorized during training.

---

## Project
Build a simple RAG app:
- [ ] Chunk a PDF/document set and embed the chunks
- [ ] Retrieve top-k relevant chunks for a user question
- [ ] Pass retrieved chunks + question to an LLM and return the answer

## Resources
- Anthropic: Prompt engineering guide — https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview
- LangChain docs (RAG patterns) — https://python.langchain.com/docs/introduction/
- Pinecone: What is RAG (concept primer) — https://www.pinecone.io/learn/retrieval-augmented-generation/
