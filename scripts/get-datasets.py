from sklearn.datasets import fetch_openml
from pathlib import Path
import pandas as pd
import openml
import json

openml.config.apikey = 'f44f61a5ad260b7dc3a501de448e47ce'

folder = Path(__file__).parents[1] / 'data/openml/'
folder.mkdir(parents=True, exist_ok=True)

to_save = folder / 'openml-datasets-CC18.csv'
if to_save.exists():
    datasets = pd.read_csv(to_save)
else:
    # Remove the `tag` keyword argument to get all datasets
    datasets = openml.datasets.list_datasets(tag='OpenML-CC18', output_format='dataframe')
    datasets.to_csv(to_save, index=False)

for _, row in datasets.iterrows():
    did = int(row['did'])
    if all(map(lambda x: Path(folder / f'{did}/{x}.csv').exists(), ['X', 'y'])):
        X = pd.read_csv(folder / f'{did}/X.csv')
        y = pd.read_csv(folder / f'{did}/y.csv')
    else:
        Path(folder / f'{did}/').mkdir(exist_ok=True, parents=True)
        X, y = fetch_openml(data_id=did, return_X_y=True, as_frame=True, parser='auto')
        X.to_csv(folder / f'{did}/X.csv', index=False)
        y.to_csv(folder / f'{did}/y.csv', index=False)
        data = fetch_openml(data_id=did, return_X_y=False, as_frame=False, parser='auto')
        description, categories, details = data['DESCR'], data['categories'], data['details']
        with open(folder / f'{did}/description.txt', 'w') as f:
            f.write(str(description))
        if categories is not None:
            X_categories = categories.copy()
            y_categories = {y.name: X_categories.pop(y.name, [])}
            with open(folder / f'{did}/X_categories.json', 'w') as f:
                json.dump(X_categories, f, ensure_ascii=False, indent=2)
            with open(folder / f'{did}/y_categories.json', 'w') as f:
                json.dump(y_categories, f, ensure_ascii=False, indent=2)
        with open(folder / f'{did}/details.json', mode="wt", encoding="utf-8") as f:
	        json.dump(details, f, ensure_ascii=False, indent=2)

    dataset = openml.datasets.get_dataset(did, download_data=True, download_qualities=True, download_features_meta_data=True)

    for k, v in dataset.features.items():
        missing = f"{v.number_missing_values} missing values" if v.number_missing_values else ""
        print(f"{v.name} ({v.data_type}), {v.nominal_values} {missing}")
    for k, q in dataset.qualities.items():
        if not pd.isna(q): print(k, q)

# Use PercentageOfInstancesWithMissingValues to screen for data with missing values
