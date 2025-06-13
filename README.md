# 🧠 ML Model Comparison Toolkit

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-brightgreen?logo=streamlit)](https://ml-toolkit-ouzbcccswpncawcypvhsz7.streamlit.app/)


A powerful, modular, and interactive dashboard for comparing machine learning models, evaluating performance, explaining predictions, and tuning hyperparameters — all in one place.

Built using **Python**, **Streamlit**, and **scikit-learn**. Designed for learning, showcasing, and real-world ML development.

---

## 🚀 Features

### 🔍 Model Comparison
- Train and evaluate: Logistic Regression, Decision Tree, Random Forest, and SVM
- Metrics: Accuracy, Precision, Recall, F1 Score, Runtime
- Interactive bar chart visualization

### 📁 Dataset Handling
- Upload your own CSV dataset
- Automatically encodes categorical features and handles missing values
- Defaults to the Iris dataset if no file is provided

### 🔬 SHAP Explainability (for Random Forest)
- Bar plots showing feature importance
- Dot plots visualizing per-instance contributions

### 📉 Performance Diagnostics
- Confusion Matrix (for binary classification)
- ROC Curve with AUC

### 🔁 Cross-Validation
- Selectable k-fold CV (2–10 folds)
- Displays mean and standard deviation for Accuracy and F1

### ⚙️ Hyperparameter Tuning
- GridSearchCV integration with preconfigured parameter grids
- Displays best parameters and cross-validated scores

---

## 🗂 Project Structure

```
ML-Toolkit/
├── app/                          # Streamlit app
│   └── dashboard.py
├── data/                         # Sample dataset
│   └── iris_sample.csv
├── notebooks/                    # Interactive exploration notebooks
│   ├── model_testing.ipynb
│   └── demo_usage.ipynb
├── src/                          # Modular ML utilities
│   ├── preprocess.py
│   ├── train_models.py
│   └── evaluate_models.py
├── .streamlit/                   # Streamlit configuration
│   └── config.toml
├── Dockerfile                    # Docker config
├── .dockerignore                 # Docker exclusions
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
├── index.html                    # GitHub Pages landing page
├── README.md                     # This file
├── project_board.md              # GitHub project tasks
├── blog_demo_toolkit.md          # Blog post: Comparing Models
├── blog_shap_explainability.md   # Blog post: SHAP Explanations
└── blog_deploy_streamlit_cloud.md # Blog post: Streamlit Deployment
```

---

## 🧪 Quickstart (Local)

```bash
# Optional: create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/dashboard.py
```

---

## 🐳 Run with Docker

```bash
docker build -t ml-toolkit .
docker run -p 8501:8501 ml-toolkit
```

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Select this repo and set the entry point to:
   ```
   app/dashboard.py
   ```
4. Add `requirements.txt` and click Deploy

---

## 📚 Blog Posts

- [🔍 Comparing ML Models](blog_demo_toolkit.md)
- [🔬 SHAP Explainability](blog_shap_explainability.md)
- [🚀 Streamlit Deployment Guide](blog_deploy_streamlit_cloud.md)

---

## 📌 Use Cases

- Educational tool for learning model selection
- Portfolio project for data science interviews
- Rapid prototyping tool for classification problems

---

## 👤 Author

**Matthew Ballard**  
Graduate Student – AI Engineering  
Davenport University  
[igobymatthew@gmail.com](mailto:igobymatthew@gmail.com)

---

## 🧠 License

This project is licensed under the [MIT License](LICENSE).  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-red)](https://streamlit.io/)
