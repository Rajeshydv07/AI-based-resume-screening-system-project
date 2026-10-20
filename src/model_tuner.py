import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def tune_models(data_dir='data', models_dir='models'):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    X_train_clean = np.clip(X_train.values, 0, None)
    X_test_clean = np.clip(X_test.values, 0, None)

    # 1. Logistic Regression search space
    lr_grid = {
        'C': [0.01, 0.1, 1.0, 5.0, 10.0],
        'solver': ['liblinear', 'lbfgs'],
        'class_weight': ['balanced', None]
    }
    lr_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid=lr_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    lr_search.fit(X_train_clean, y_train)

    # 2. Random Forest search space
    rf_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }
    rf_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=rf_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    rf_search.fit(X_train_clean, y_train)

    # 3. SVM search space
    svm_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear'],
        'class_weight': ['balanced', None]
    }
    svm_search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid=svm_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    svm_search.fit(X_train_clean, y_train)

    tuned_candidates = [
        ('Logistic Regression', lr_search.best_estimator_, lr_search.best_params_, lr_search.best_score_),
        ('Random Forest', rf_search.best_estimator_, rf_search.best_params_, rf_search.best_score_),
        ('Support Vector Machine', svm_search.best_estimator_, svm_search.best_params_, svm_search.best_score_)
    ]

    records = []
    best_overall_score = -1
    best_overall_model = None
    best_overall_name = ""

    for name, model, params, best_cv_f1 in tuned_candidates:
        y_pred = model.predict(X_test_clean)
        y_prob = model.predict_proba(X_test_clean)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        records.append({
            'Model': name,
            'Best Parameters': str(params),
            'Tuned CV F1': round(float(best_cv_f1), 4),
            'Test Accuracy': round(float(acc), 4),
            'Test F1': round(float(f1), 4),
            'Test Precision': round(float(prec), 4),
            'Test Recall': round(float(rec), 4),
            'Test ROC-AUC': round(float(auc), 4)
        })

        if f1 > best_overall_score:
            best_overall_score = f1
            best_overall_model = model
            best_overall_name = name

    df_tuned = pd.DataFrame(records)
    
    # save tuned results and best model
    os.makedirs(models_dir, exist_ok=True)
    df_tuned.to_csv(os.path.join(models_dir, 'tuning_results.csv'), index=False)
    with open(os.path.join(models_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(best_overall_model, f)
    with open(os.path.join(models_dir, 'best_model_info.pkl'), 'wb') as f:
        pickle.dump({'name': best_overall_name, 'score': best_overall_score}, f)

    return df_tuned, best_overall_name

def main():
    print("Starting hyperparameter tuning...")
    df_tuned, best_name = tune_models()
    print("Tuning Complete! Best Model:", best_name)
    print(df_tuned.to_string(index=False))

if __name__ == '__main__':
    main()
