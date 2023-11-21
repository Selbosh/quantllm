from pathlib import Path
from jenga.tasks.openml import OpenMLRegressionTask, OpenMLBinaryClassificationTask, OpenMLMultiClassClassificationTask
from jenga.corruptions.generic import MissingValues
from jenga.evaluation.corruption_impact import CorruptionImpactEvaluator
import matplotlib.pyplot as plt
import numpy as np


class ModeImputer:
    def __init__(self, columns):
        self.columns = columns
        self.modes = {}

    def fit(self, data):
        for column in self.columns:
            mode = data[column].value_counts().index[0]
            self.modes[column] = mode

    def transform(self, data):
        imputed = data.copy(deep=True)
        for column in self.columns:
            imputed[column].fillna(self.modes[column], inplace=True) 
        return imputed


class ChainedModelDecorator:
    def __init__(self, model, imputers):
        self.model = model
        self.imputers = imputers

    def predict_proba(self, data):
        imputed = data
        for imputer in self.imputers:
            imputed = imputer.transform(imputed)

        return self.model.predict_proba(imputed)


class ModelDecorator:
    def __init__(self, model, imputer):
        self.model = model
        self.imputer = imputer

    def predict_proba(self, data):
        return self.model.predict_proba(self.imputer.transform(data))


def find_result(column, fraction, missingness, results):
    for result in results:
        corr = result.corruption
        if corr.column == column and corr.fraction == fraction and corr.sampling == missingness:
            return result


def plot_impact(column, plt, results, suffix=''):
    ax = plt.gca()

    scores = []
    labels = []

    for impacted_column in [column]:
        for fraction in [0.01, 0.1, 0.5, 0.99]:  
            for missingness in ['MNAR', 'MAR', 'MCAR']:                    
                result = find_result(impacted_column, fraction, missingness, results)
                scores.append(result.corrupted_scores)
                labels.append(f"{missingness} {int(fraction*100)}%")

    baseline_score = result.baseline_score            

    ax.axhline(baseline_score, linestyle='--', color='red')
    bplot = ax.boxplot(scores, showfliers=False, patch_artist=True, medianprops={'color':'black'})

    colors = [
        '#1e4052', '#dc6082', '#e1a677',
        '#1e4052', '#dc6082', '#e1a677', 
        '#1e4052', '#dc6082', '#e1a677', 
        '#1e4052', '#dc6082', '#e1a677'
    ]

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticks(np.arange(1, len(labels) + 1))

    ax.yaxis.grid(True)
    ax.set_xticklabels(labels)

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    ax.set_title(f"Missing values in '{column}'", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)    
    ax.set_ylabel('AUC', fontsize=24)

    plt.gcf().set_size_inches(8, 6)
    plt.tight_layout()
    
    dir = Path(__file__).parents[1] / 'data' / 'output' / 'imputation' / 'baseline'
    dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(dir / f'{column}{suffix}.pdf', bbox_inches='tight')


def main():
    openml_id = 40983
    task = OpenMLBinaryClassificationTask(openml_id=openml_id, train_size=0.8, seed=42)
    model = task.fit_baseline_model()
    evaluator = CorruptionImpactEvaluator(task)

    columns = task.categorical_columns + task.numerical_columns

    corruptions = []
    for column in columns:
        for fraction in [0.01, 0.1, 0.5, 0.99]:
            for missingness in ['MNAR', 'MAR', 'MCAR']:
                corruption = MissingValues(column=column, fraction=fraction, missingness=missingness)
                corruptions.append(corruption)

    imputer = ModeImputer(columns)
    imputer.fit(task.train_data)
    
    mode_model = ModelDecorator(model, imputer)
    results = evaluator.evaluate(model, 10, *corruptions)
    

    for column in columns:
        plot_impact(column, plt, results)


if __name__ == "__main__":
    main()
