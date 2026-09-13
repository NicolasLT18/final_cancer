# Bayesian Network for Disease Diagnosis from Symptoms

Ángel Nicolás Landa Tapia - A01668133
Ximena Paola Pérez Sánchez - A01782269

A Naive Bayes bayesian network (`Disease` → symptoms) built with `pgmpy` that, given a set of observed symptoms, infers a posterior probability distribution over possible diseases using exact inference.

## Dataset

[Disease Symptom Prediction](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) (Kaggle, `itachi9604/disease-symptom-description-dataset`).

Files used (expected under `data/`):
- `dataset.csv`: 4920 entries, each with a disease and up to 17 symptoms.
- `Symptom-severity.csv`: severity weight of each individual symptom.
- `symptom_Description.csv`: text description of each disease.
- `symptom_precaution.csv`: up to 4 recommended precautions per disease.

## Contents

The notebook ([modelo.ipynb](modelo.ipynb)) is organized into:
1. Preprocessing (text normalization, one-hot encoding of symptoms)
2. Exploratory analysis (symptom frequency, discriminative power per disease)
3. Bayesian network construction (Naive Bayes structure, CPD estimation, comparison against Hill-Climb Search)
4. Inference (querying the posterior over `Disease` given partial symptom evidence)
5. Evaluation (accuracy/classification report, plus a symptom-combination-grouped split to rule out data leakage)
6. Final summary (results, limitations, possible improvements)

## Setup

```bash
pip install -r requirements.txt
jupyter notebook modelo.ipynb
```

The dataset CSVs must be placed in a `data/` folder next to the notebook before running it.
