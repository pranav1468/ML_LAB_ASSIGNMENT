import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from src.ML_LAB.utils.common import save_json
from src.ML_LAB.entity.config_entity import *
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        # Classification Metrics
        acc = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted') # weighted handles imbalance if any
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        return acc, precision, recall, f1

    def save_results(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]
        
        predicted_qualities = model.predict(test_x)

        (acc, precision, recall, f1) = self.eval_metrics(test_y, predicted_qualities)
        
        # Saving metrics as local JSON
        scores = {
            "accuracy": acc, 
            "precision": precision, 
            "recall": recall, 
            "f1_score": f1
        }
        
        save_json(path=Path(self.config.metric_file_name), data=scores)