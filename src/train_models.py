from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'LightGBM': LGBMClassifier()
    }
