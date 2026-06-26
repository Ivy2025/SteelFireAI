# SteelFireAI

SteelFireAI is an open-source machine learning framework for predicting the elevated-temperature mechanical properties of structural steels from their chemical composition, manufacturing process, and temperature.

This repository accompanies the research on data-driven prediction of steel mechanical properties under fire conditions and provides the source code for model development, evaluation, explainability analysis, and GUI deployment.

---

## Repository Structure

```text
SteelFireAI/
├── data/              Experimental datasets
├── scripts/           Data processing and model training
├── models/            Trained machine learning models
├── gui/               GUI application
├── shap/              SHAP analysis
├── requirements.txt
└── README.md
```

---

## Methodology

The workflow of SteelFireAI consists of:

1. Data collection and preprocessing
2. Feature engineering
3. Steel-grade grouped train/test splitting
4. Machine learning model training
5. Model evaluation
6. SHAP explainability analysis
7. GUI deployment

The implemented machine learning models include:

- Random Forest
- XGBoost
- LightGBM
- CatBoost
- DT
- SVR
- KNN

---

## Installation

Clone this repository

```bash
git clone https://github.com/Ivy2025/SteelFireAI.git
cd SteelFireAI
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## Usage

Train the models

```bash
python train.py
```

Launch the GUI

```bash
python gui.py
```

Validate the models

```bash
python validation_prediction.py
```
---

## Dataset

The dataset was compiled from published experimental studies on the elevated-temperature mechanical properties of structural steels.

Each sample contains:

- Chemical composition
- Manufacturing process
- Temperature
- Yield strength reduction factor
- Ultimate strength reduction factor
- Elastic modulus reduction factor

The dataset used in this study is available from the corresponding author upon reasonable request.

---


