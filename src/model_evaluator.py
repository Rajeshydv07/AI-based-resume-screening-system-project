import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

def evaluate_best_model(data_dir='data', models_dir='models'):
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    with open(os.path.join(models_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
        
    with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)

    X_test_clean = np.clip(X_test.values, 0, None)
    
    # default threshold 0.5
    y_probs = model.predict_proba(X_test_clean)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test_clean)
    y_preds = (y_probs >= 0.5).astype(int)
    
    metrics = {
        'Accuracy': round(float(accuracy_score(y_test, y_preds)), 4),
        'Precision': round(float(precision_score(y_test, y_preds)), 4),
        'Recall': round(float(recall_score(y_test, y_preds)), 4),
        'F1 Score': round(float(f1_score(y_test, y_preds)), 4),
        'ROC-AUC': round(float(roc_auc_score(y_test, y_probs)), 4)
    }
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_preds).tolist()
    
    # roc curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
    
    # precision recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_probs)
    pr_data = {'precision': prec.tolist(), 'recall': rec.tolist()}
    
    # feature importance
    importance = []
    if hasattr(model, 'feature_importances_'):
        imp_vals = model.feature_importances_
        for name, val in zip(feature_names, imp_vals):
            importance.append({'Feature': name, 'Importance': float(val)})
    elif hasattr(model, 'coef_'):
        coef_vals = np.abs(model.coef_[0])
        for name, val in zip(feature_names, coef_vals):
            importance.append({'Feature': name, 'Importance': float(val)})
    else:
        # for non-linear rbf kernel SVM, use absolute correlation with y
        for i, name in enumerate(feature_names):
            corr = np.corrcoef(X_test_clean[:, i], y_test)[0, 1]
            importance.append({'Feature': name, 'Importance': float(abs(corr) if not np.isnan(corr) else 0.0)})
            
    imp_df = pd.DataFrame(importance).sort_values(by='Importance', ascending=False)
    
    eval_results = {
        'metrics': metrics,
        'confusion_matrix': cm,
        'roc_curve': roc_data,
        'pr_curve': pr_data,
        'top_features': imp_df.head(20).to_dict(orient='records')
    }
    
    with open(os.path.join(models_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump(eval_results, f)
        
    return eval_results

def main():
    print("Evaluating best model...")
    eval_res = evaluate_best_model()
    print("Metrics:", eval_res['metrics'])

if __name__ == '__main__':
    main()
