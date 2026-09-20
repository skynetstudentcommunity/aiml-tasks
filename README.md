# AI/ML Learning Roadmap 🚀

A 16-week, hands-on path from **zero to job-ready** in Machine Learning & AI. Each week has a clear goal, concrete daily-sized tasks, and a small deliverable you push to this repo — so by the end you have a portfolio, not just notes.

**Time commitment:** ~8–10 hrs/week (can be compressed or stretched)
**Prerequisite:** Basic Python syntax (variables, loops, functions)

## How to use this repo
- Create one folder per week: `week-01/`, `week-02/`, etc.
- Each folder should contain your code/notebook + a short `notes.md`
- Check off tasks below as you complete them
- Don't skip the weekly project — it's where the actual learning happens

---

## Phase 1: Foundations (Weeks 1–4)

### Week 1 — Python for Data
**Goal:** Get comfortable with the tools you'll use every week after this.
- [ ] Set up Python, VS Code/Jupyter, and a virtual environment
- [ ] Learn NumPy: arrays, indexing, broadcasting, vectorized ops
- [ ] Learn Pandas: DataFrames, filtering, groupby, merging
- [ ] Learn Matplotlib/Seaborn: line, bar, scatter, histogram plots
- **Project:** Load a public CSV dataset (e.g., Titanic or a dataset of your choice) and produce a short EDA (exploratory data analysis) notebook with 5+ visualizations and written observations.

### Week 2 — Math You Actually Need
**Goal:** Build just enough math intuition to understand *why* ML algorithms work.
- [ ] Linear algebra: vectors, matrices, dot product, matrix multiplication
- [ ] Calculus: derivatives, gradients, chain rule (conceptual, not proofs)
- [ ] Probability & statistics: mean/variance, distributions, Bayes' theorem
- [ ] Implement gradient descent from scratch on a simple function (e.g., minimize `f(x) = x²`)
- **Project:** Write a from-scratch linear regression (no sklearn) using gradient descent, and plot the loss curve over iterations.

### Week 3 — Data Wrangling & Feature Engineering
**Goal:** Learn to turn messy real-world data into model-ready data.
- [ ] Handling missing values (drop vs. impute)
- [ ] Encoding categorical variables (one-hot, label encoding)
- [ ] Feature scaling (normalization vs. standardization)
- [ ] Train/validation/test splits, and why they matter
- **Project:** Take a messy real dataset (Kaggle "messy" datasets work well) and produce a full cleaning pipeline script.

### Week 4 — Core ML Concepts
**Goal:** Understand the vocabulary and workflow before touching more algorithms.
- [ ] Supervised vs. unsupervised vs. reinforcement learning
- [ ] Overfitting vs. underfitting, bias-variance tradeoff
- [ ] Evaluation metrics: accuracy, precision, recall, F1, RMSE, R²
- [ ] Cross-validation
- **Project:** Write a one-page cheat sheet (in your own words) explaining all the above terms with examples — you'll refer back to this constantly.

---

## Phase 2: Core Machine Learning (Weeks 5–8)

### Week 5 — Regression & Classification with Scikit-learn
- [ ] Linear & Logistic Regression (using sklearn this time)
- [ ] K-Nearest Neighbors
- [ ] Decision Trees
- [ ] Confusion matrix and ROC curves
- **Project:** Build a classifier (e.g., predict loan default / disease diagnosis) and compare 3 models on the same dataset using a metrics table.

### Week 6 — Ensemble Methods
- [ ] Random Forests
- [ ] Gradient Boosting (XGBoost or LightGBM)
- [ ] Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- [ ] Feature importance
- **Project:** Take last week's dataset and beat your previous best score using an ensemble model + tuning. Document the improvement.

### Week 7 — Unsupervised Learning
- [ ] K-Means clustering
- [ ] Hierarchical clustering
- [ ] PCA (dimensionality reduction)
- [ ] DBSCAN (optional, for density-based clustering)
- **Project:** Cluster a customer/product dataset and visualize the clusters in 2D using PCA. Write a short interpretation of each cluster.

### Week 8 — ML Project Week (Consolidation)
**Goal:** No new theory — apply everything from Weeks 1–7 end-to-end.
- [ ] Pick a Kaggle competition (or a real dataset you care about)
- [ ] Full pipeline: EDA → cleaning → feature engineering → model comparison → tuning
- [ ] Write a clear README for this specific project
- **Project:** A polished, portfolio-ready classical ML project pushed to its own repo (or a `capstone-1/` folder here).

---

## Phase 3: Deep Learning (Weeks 9–12)

### Week 9 — Neural Network Fundamentals
- [ ] Perceptrons, activation functions (ReLU, sigmoid, softmax)
- [ ] Forward propagation & backpropagation (conceptual + by hand on paper)
- [ ] Loss functions and optimizers (SGD, Adam)
- [ ] Build a neural net from scratch using only NumPy (no frameworks)
- **Project:** Train your from-scratch NN on MNIST digits and report accuracy.

### Week 10 — PyTorch or TensorFlow (pick one, stick with it)
- [ ] Tensors, autograd, building `nn.Module` models
- [ ] Training loops, batching, DataLoaders
- [ ] Regularization: dropout, batch norm, early stopping
- **Project:** Rebuild the MNIST classifier in your chosen framework and improve accuracy above your from-scratch version.

### Week 11 — Convolutional Neural Networks (Computer Vision)
- [ ] Convolutions, pooling, CNN architectures (LeNet, ResNet basics)
- [ ] Data augmentation
- [ ] Transfer learning (fine-tune a pretrained model like ResNet18)
- **Project:** Build an image classifier for a custom dataset (e.g., your own photos, or a Kaggle image dataset) using transfer learning.

### Week 12 — Sequence Models (NLP Intro)
- [ ] Text preprocessing: tokenization, embeddings (Word2Vec/GloVe concepts)
- [ ] RNNs/LSTMs (conceptual understanding)
- [ ] Intro to Transformers and attention (conceptual — full depth comes later)
- **Project:** Build a sentiment analysis model (movie reviews or tweets) using embeddings + a simple neural classifier.

---

## Phase 4: Advanced Topics & Deployment (Weeks 13–16)

### Week 13 — Transformers & Modern NLP
- [ ] Attention mechanism in depth
- [ ] Using HuggingFace `transformers` library
- [ ] Fine-tuning a pretrained model (BERT/DistilBERT) for classification
- **Project:** Fine-tune a pretrained transformer on a text classification task and compare against your Week 12 model.

### Week 14 — Generative AI & LLMs
- [ ] How LLMs work at a high level (tokenization, next-token prediction)
- [ ] Prompt engineering basics
- [ ] Using an LLM API to build a small tool (e.g., summarizer, Q&A bot)
- [ ] Intro to RAG (Retrieval-Augmented Generation)
- **Project:** Build a simple RAG app: feed it a PDF/document set and let it answer questions grounded in that content.

### Week 15 — MLOps & Deployment
- [ ] Model serialization (pickle, ONNX, torch.save)
- [ ] Building an API around your model (FastAPI/Flask)
- [ ] Containerizing with Docker
- [ ] Deploying to a free-tier cloud service (Render, HuggingFace Spaces, or Streamlit Cloud)
- **Project:** Take any earlier model and deploy it as a live, shareable web app with a simple UI.

### Week 16 — Capstone Project
**Goal:** Build something end-to-end that demonstrates everything you've learned.
- [ ] Choose a real problem (not a toy dataset) that interests you
- [ ] Full pipeline: data collection/cleaning → modeling → evaluation → deployment
- [ ] Write a proper README with problem statement, approach, results, and a demo link/GIF
- **Project:** Your flagship portfolio piece. This is what you link on your resume/LinkedIn.

---

## Suggested Resources (optional, pick what fits your style)
- **Courses:** Andrew Ng's ML Specialization, fast.ai Practical Deep Learning
- **Books:** *Hands-On Machine Learning* (Géron), *Deep Learning* (Goodfellow) for reference
- **Practice:** Kaggle competitions & datasets
- **Docs:** scikit-learn, PyTorch, HuggingFace official documentation (best source of truth)

## Progress Tracker

| Phase | Weeks | Status |
|---|---|---|
| Foundations | 1–4 | ☐ |
| Core ML | 5–8 | ☐ |
| Deep Learning | 9–12 | ☐ |
| Advanced & Deployment | 13–16 | ☐ |

---

*Update this README as you go — check off tasks, link your notebooks, and add screenshots of results. A README that shows visible progress is itself part of the portfolio.*
