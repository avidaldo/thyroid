import json

path = '/media/DIURNOext4/alejandro/wip-clase/PIA-SAA/example_repos/thyroid/03_advanced_models.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell.get('source', []):
            if "cat_features=cat_feature_indices" in line and "CatBoostClassifier" not in line:
                continue # remove this line (it's inside the CatBoostClassifier init)
            new_source.append(line)
        cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

