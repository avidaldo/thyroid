import json
import os

repo_dir = '/media/DIURNOext4/alejandro/wip-clase/PIA-SAA/example_repos/thyroid'
path = os.path.join(repo_dir, '03_advanced_models.ipynb')

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        
        # Restore cat_features to constructor in cell 42 if we removed it
        if "catboost_model = CatBoostClassifier(" in source and "cat_features" not in source:
            source = source.replace("depth=6,\n", "depth=6,\n    cat_features=cat_feature_indices,\n")
            cell['source'] = [line + '\n' for line in source.split('\n')[:-1]]
            
        # Rewrite cross_val_score to manual loop
        original_cv = "scores_catboost = cross_val_score("
        if original_cv in source:
            manual_cv = """import numpy as np
scores_catboost = []
for train_idx, val_idx in stratified_cv.split(X_train_cat, y_train):
    cb = CatBoostClassifier(
        iterations=100, learning_rate=0.1, depth=6, 
        cat_features=cat_feature_indices, random_state=42, verbose=False
    )
    cb.fit(X_train_cat.iloc[train_idx], y_train.iloc[train_idx])
    score = thyroid_scorer(cb, X_train_cat.iloc[val_idx], y_train.iloc[val_idx])
    scores_catboost.append(score)
scores_catboost = np.array(scores_catboost)

print(f"CatBoost - Thyroid Mean Recall")
print(f"  Per-fold scores: {scores_catboost.round(3)}")
print(f"  Mean: {scores_catboost.mean():.3f} (+/- {scores_catboost.std() * 2:.3f})")
"""
            cell['source'] = [line + '\n' for line in manual_cv.split('\n')[:-1]]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
