# Week 8: ML Project Week — Tutorial

## Why this week matters
No new theory this week. The skill that actually gets you hired isn't
knowing individual algorithms — it's stitching them into a complete,
working pipeline on a real problem. This week is that dress rehearsal.

---

## 1. The Full Pipeline, End to End

Raw data → EDA → Cleaning → Feature Engineering →
Train/Val/Test Split → Model Comparison → Tuning → Final Evaluation → README


Each arrow is a week you already did — this week is about doing all of them
back-to-back on one dataset, without a tutorial telling you each next step.

---

## 2. Picking a Good Project Dataset

Look for something with:
- A clear target variable (what are you predicting?)
- Enough rows to be meaningful (thousands, not dozens)
- Some real-world messiness (so cleaning skills actually get used)
- Ideally something you have *some* interest in — you'll look at it a lot

Kaggle's "Getting Started" competitions are a good source, or use a public
dataset in a domain you care about (sports, finance, health, etc.).

---

## 3. Structuring Your Repo

capstone-1/
├── data/ # raw + processed data (or a script to fetch it)
├── notebooks/ # exploration
├── src/ # cleaning.py, features.py, train.py
├── models/ # saved model files
└── README.md


Keeping exploration (notebooks) separate from reusable code (src/) is a
habit worth building now — it pays off hugely from Week 15 onward.

---

## 4. Writing the README

A good project README, at minimum, covers:
- **Problem**: what are you predicting and why does it matter?
- **Data**: source, size, key features
- **Approach**: models tried, why
- **Results**: a metrics table, with your best model highlighted
- **What you'd do next**: shows you know the work isn't "done," it's just
  where you stopped

---

## Project
- [ ] Pick a dataset/competition
- [ ] Run the full pipeline: EDA → cleaning → feature engineering →
      model comparison → tuning
- [ ] Write the README as described above
- [ ] Push to its own repo or a `capstone-1/` folder

## Resources
- Kaggle Competitions — https://www.kaggle.com/competitions
- Made With ML (production-quality ML practices) — https://madewithml.com/
- Awesome README templates — https://github.com/othneildrew/Best-README-Template
