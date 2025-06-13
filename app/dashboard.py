import streamlit as st
import pandas as pd
import numpy as np
import time
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

st.title("ML Model Comparison Toolkit")
st.markdown("Compare, evaluate, and tune machine learning models interactively.")

# Upload or load data
# Select a sample dataset or upload your own
sample_datasets = {
    "Iris (default)": "data/iris_sample.csv",
    "Titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "Wine": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/wine.csv"
}
selected_dataset = st.selectbox("📂 Choose a sample dataset or upload your own:", list(sample_datasets.keys()))

uploaded_file = st.file_uploader("Or upload your own CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Dataset Preview:")
    st.dataframe(df.head())
    target_column = st.selectbox("🎯 Select the target column", options=df.columns)
    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X = pd.get_dummies(X)
        y = pd.factorize(y)[0]
        st.success(f"Using uploaded dataset with shape {X.shape}")
else:
    dataset_path = sample_datasets[selected_dataset]
    df = pd.read_csv(dataset_path)
    if selected_dataset == "Iris (default)":
        target_column = "species"
    elif selected_dataset == "Titanic":
        target_column = "Survived"
    elif selected_dataset == "Wine":
        target_column = "quality" if "quality" in df.columns else df.columns[-1]

    st.info(f"Using sample dataset: {selected_dataset}")
    st.dataframe(df.head())
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = pd.get_dummies(X)
    y = pd.factorize(y)[0]
    st.success(f"Using uploaded dataset with shape {X.shape}")
else:
    df = pd.read_csv("data/iris_sample.csv")
    target_column = "species"
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = pd.get_dummies(X)
    y = pd.factorize(y)[0]
    st.info("Using built-in sample dataset (Iris)")

if len(df) < 5:
    st.error("Dataset is too small to split into training and test sets. Please upload more data.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_options = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

selected_models = st.multiselect("Select models to compare:", options=list(model_options.keys()), default=list(model_options.keys()))

results = []
if selected_models:
    for name in selected_models:
        model = model_options[name]
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed_time = time.time() - start_time

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'Runtime (s)': elapsed_time
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df.style.format({
        "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}",
        "F1 Score": "{:.2%}", "Runtime (s)": "{:.4f}"
    }))

    metric_to_plot = st.selectbox("Choose metric to plot:", ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Runtime (s)'])
    st.bar_chart(data=results_df.set_index("Model")[[metric_to_plot]])
else:
    st.warning("Please select at least one model to compare.")

# SHAP Explainability
st.subheader("SHAP Explainability (Tree-Based Models Only)")
if 'Random Forest' in selected_models:
    try:
        explainer = shap.TreeExplainer(model_options['Random Forest'])
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            for i, class_values in enumerate(shap_values):
                try:
                    st.markdown(f"**SHAP Summary for Class {i}**")
                    shap.summary_plot(class_values, X_test, show=False)
                    fig = plt.gcf()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not plot SHAP for class {i}: {str(e)}")
        else:
            shap.summary_plot(shap_values, X_test, show=False)
            fig = plt.gcf()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"SHAP could not be generated: {str(e)}")
else:
    st.info("Select 'Random Forest' to view SHAP explanations.")

# Confusion Matrix & ROC
st.subheader("Confusion Matrix & ROC Curve (Binary or Simplified Classes Only)")
if len(set(y_test)) == 2:
    for name in selected_models:
        model = model_options[name]
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        y_pred = model.predict(X_test)

        st.markdown(f"**Model: {name}**")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
        disp.plot(ax=ax, cmap="Blues")
        st.pyplot(fig)

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
            ax_roc.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title(f"ROC Curve - {name}")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
else:
    st.info("ROC curve & confusion matrix shown only for binary classification.")

# Cross-validation
st.subheader("Cross-Validation Summary")
cv_results = []
cv_folds = st.slider("Select number of folds for cross-validation", min_value=2, max_value=10, value=5)
for name in selected_models:
    model = model_options[name]
    try:
        acc_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        f1_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='f1_macro')
        cv_results.append({
            "Model": name,
            "CV Accuracy (mean)": np.mean(acc_scores),
            "CV Accuracy (std)": np.std(acc_scores),
            "CV F1 Score (mean)": np.mean(f1_scores),
            "CV F1 Score (std)": np.std(f1_scores),
        })
    except Exception as e:
        st.warning(f"{name} cross-validation failed: {e}")

if cv_results:
    df_cv = pd.DataFrame(cv_results)
    if not df_cv.empty:
        st.dataframe(df_cv.set_index("Model").style.format({
            "CV Accuracy (mean)": "{:.2%}",
            "CV Accuracy (std)": "{:.2%}",
            "CV F1 Score (mean)": "{:.2%}",
            "CV F1 Score (std)": "{:.2%}"
        }), use_container_width=True)
    else:
        st.info("No cross-validation results to display.")

# GridSearchCV Tuning
st.subheader("Hyperparameter Tuning (GridSearchCV)")
grid_results = []
param_grids = {
    "Logistic Regression": {"C": [0.1, 1.0, 10.0]},
    "Decision Tree": {"max_depth": [3, 5, 10]},
    "Random Forest": {"n_estimators": [50, 100], "max_depth": [3, 5, None]},
    "SVM": {"C": [0.1, 1.0, 10.0], "kernel": ["linear", "rbf"]}
}

for name in selected_models:
    if name in param_grids:
        model = model_options[name]
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring='accuracy', n_jobs=-1)
        try:
            grid.fit(X_train, y_train)
            grid_results.append({
                "Model": name,
                "Best Params": str(grid.best_params_),
                "Best CV Accuracy": grid.best_score_
            })
        except Exception as e:
            st.warning(f"{name} tuning failed: {e}")

if grid_results:
    df_grid = pd.DataFrame(grid_results)
    if not df_grid.empty:
        st.dataframe(df_grid.set_index("Model").style.format({
            "Best CV Accuracy": "{:.2%}"
        }), use_container_width=True)
    else:
        st.info("No GridSearchCV results to display.")
