# AI Classifier Lab - Iris Flower Classification

**Project:** AI_Classifier_Lab  
**Group:** GroupAlpha  
**Type:** Interactive ML Classification with Streamlit

---

## Overview

This project is a complete, production-ready machine learning application that classifies Iris flowers into three species (setosa, versicolor, virginica) based on four physical measurements. It features an interactive Streamlit web interface with real-time model training, evaluation metrics, visualization charts, and a comparison module for testing different configurations.

---

## Project Structure

```
AI_Classifier_Lab/
├── app.py                  # Main Streamlit entry file
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── data/
    └── dataset.csv        # Iris flower classification dataset
```

---

## Features

- **Problem Setup Module:** Upload your own dataset or use the built-in Iris dataset
- **Visual UI Module:** Interactive sliders, dropdowns, and real-time controls
- **Core ML Logic:** Train Logistic Regression or Random Forest with configurable hyperparameters
- **Evaluation Module:** Accuracy, Precision, and Recall metrics with confusion matrix
- **Comparison Module:** Save and compare up to 5 different experiment runs
- **Explainability:** Natural-language explanations and feature importance analysis
- **Visualization:** Interactive Plotly charts (scatter plots, distributions, confusion matrix)

---

## Setup Instructions

### 1. Clone or Download the Project

```bash
cd AI_Classifier_Lab/GroupAlpha
```

### 2. Create a Virtual Environment (Recommended)

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

---

## How to Use

1. **Configure your experiment** using the sidebar controls:
   - Select algorithm (Logistic Regression or Random Forest)
   - Adjust test/train split ratio
   - Tune hyperparameters (max_depth, n_estimators, regularization)

2. **Click "Run Experiment"** to train the model and see results

3. **View Results:**
   - Performance metrics (Accuracy, Precision, Recall)
   - Confusion matrix heatmap
   - Feature importance chart
   - Natural language explanation

4. **Compare Runs:** Save up to 5 experiments and compare them side-by-side

---

## Dataset

The default dataset is the classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) containing 150 samples:

| Feature | Description |
|---------|-------------|
| sepal_length | Sepal length in cm |
| sepal_width | Sepal width in cm |
| petal_length | Petal length in cm |
| petal_width | Petal width in cm |
| species | Target class (setosa, versicolor, virginica) |

You can also upload your own CSV file with the same structure.

---

## Deployment to Streamlit Community Cloud

See the deployment instructions in the main project documentation or follow the quick steps:

1. Push this folder to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path as `app.py`
5. Click Deploy

---

## Requirements

- Python 3.10+
- Streamlit 1.40+
- scikit-learn 1.5+
- pandas 2.2+
- numpy 2.1+
- plotly 5.24+

---

## Authors

GroupAlpha - AI Lab Project