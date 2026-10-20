import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }

def train_baseline_models(data_dir='data', models_dir='models'):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()

    # Naive Bayes requires non-negative values; ensure X is non-negative
    # our scaled features are minmax [0,1], binary dummies [0,1], tfidf >= 0
    # ensure any negative rounding is clipped at 0
    X_train_clean = np.clip(X_train.values, 0, None)
    X_test_clean = np.clip(X_test.values, 0, None)

    models = get_models()
    results = []
    trained_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        start_time = time.time()
        
        # 5-fold cross validation
        cv_scores = cross_validate(
            model,
            X_train_clean,
            y_train,
            cv=cv,
            scoring=['accuracy', 'f1', 'precision', 'recall']
        )
        train_time = time.time() - start_time

        # fit on full training set
        model.fit(X_train_clean, y_train)
        y_pred = model.predict(X_test_clean)
        
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred)
        test_rec = recall_score(y_test, y_pred)

        results.append({
            'Model': name,
            'CV Accuracy': round(float(np.mean(cv_scores['test_accuracy'])), 4),
            'CV F1': round(float(np.mean(cv_scores['test_f1'])), 4),
            'Test Accuracy': round(float(test_acc), 4),
            'Test F1': round(float(test_f1), 4),
            'Test Precision': round(float(test_prec), 4),
            'Test Recall': round(float(test_rec), 4),
            'Train Time (s)': round(train_time, 2)
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results)
    
    # save results and baseline models
    os.makedirs(models_dir, exist_ok=True)
    results_df.to_csv(os.path.join(models_dir, 'baseline_results.csv'), index=False)
    with open(os.path.join(models_dir, 'baseline_models.pkl'), 'wb') as f:
        pickle.dump(trained_models, f)

    return results_df

def main():
    print("Training baseline models...")
    res = train_baseline_models()
    print("Baseline Training Results:")
    print(res.to_string(index=False))

if __name__ == '__main__':
    main()
