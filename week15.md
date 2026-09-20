# Week 15: MLOps & Deployment — Tutorial

## Why this week matters
A model in a notebook helps no one. This week turns a trained model into
something other people (or your own frontend) can actually call.

---

## 1. Model Serialization

```python
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load later:
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# PyTorch equivalent:
import torch
torch.save(model.state_dict(), "model.pt")
```

Saving lets you decouple training (slow, done once) from serving
(fast, done repeatedly).

---

## 2. Building an API with FastAPI

```python
from fastapi import FastAPI
import pickle

app = FastAPI()
model = pickle.load(open("model.pkl", "rb"))

@app.post("/predict")
def predict(features: dict):
    x = [[features["age"], features["income"]]]
    prediction = model.predict(x)
    return {"prediction": prediction.tolist()}
```

Run locally with `uvicorn main:app --reload`, then test at
`http://localhost:8000/docs` (FastAPI auto-generates an interactive UI).

---

## 3. Containerizing with Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t my-model-api .
docker run -p 8000:8000 my-model-api
```

Docker packages your code + dependencies + runtime into one portable image —
"it works on my machine" becomes "it works everywhere this image runs."

---

## 4. Deploying

Free-tier options to get a live URL without managing servers:
- **Render**: deploys a Dockerized FastAPI app directly from a GitHub repo
- **HuggingFace Spaces**: great for Gradio/Streamlit demos, especially for
  ML models
- **Streamlit Cloud**: fastest path if your UI is built in Streamlit

---

## Project
Deploy an earlier model as a live, shareable web app:
- [ ] Serialize the model
- [ ] Wrap it in a FastAPI (or Flask) endpoint
- [ ] Containerize with Docker
- [ ] Deploy to Render, HuggingFace Spaces, or Streamlit Cloud and share the link

## Resources
- FastAPI docs — https://fastapi.tiangolo.com/
- Docker: Get Started guide — https://docs.docker.com/get-started/
- HuggingFace Spaces docs — https://huggingface.co/docs/hub/spaces
