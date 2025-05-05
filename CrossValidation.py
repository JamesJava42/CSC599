from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

def k_fold_cross_validate(model, X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append({
            'accuracy': accuracy_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred, average='weighted')
        })
    return scores
