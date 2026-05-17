# Master Thesis Code: Persian Financial Sentiment Analysis using ParsBERT and Polars

## Overview
This repository contains the data engineering and machine learning pipeline developed for my Master's thesis on the impact of social networks on financial markets. It implements an end-to-end Natural Language Processing (NLP) workflow that transforms raw, unstructured financial message board data into analysis-ready, balanced econometric panels.

## Methodological Highlights
* **Transformer-Based Inference:** Utilizes a fine-tuned ParsBERT model (`transformers`, `PyTorch`) for signed sentiment scoring, alongside a robust TF-IDF/Logistic Regression baseline employing distant supervision via financial lexicons.
* **Econometric Panel Construction:** Uses `polars` for high-performance data manipulation, reshaping wide message arrays into long formats, and collapsing them into daily (ticker-date) panels.
* **Data Integrity for Causal Inference:** Strictly handles unbalanced calendar grids (imputing true zeros for days with no messages rather than dropping them) to ensure validity in downstream fixed-effects estimation and event studies.
* **Advanced Text Preprocessing:** Implements custom character unification, ZWNJ handling, and negation-scope tagging adapted for Persian financial NLP.

## Tech Stack
* **Data Processing:** `polars`, `numpy`
* **Machine Learning:** `transformers`, `torch`, `scikit-learn`
* **NLP:** `hazm`
* **Visualization:** `matplotlib` (with Persian RTL text support)
