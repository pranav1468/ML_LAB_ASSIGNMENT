import pandas as pd
import os
from src.ML_LAB import logger
from sklearn.ensemble import RandomForestClassifier # Changed import
import joblib
from src.ML_LAB.entity.config_entity import *

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # 1. Load Data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # 2. Split X and y
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)

        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        # 3. Initialize Random Forest with params from config
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            criterion=self.config.criterion,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state
        )

        # 4. Train Model
        rf.fit(train_x, train_y)

        # 5. Save Model
        save_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(rf, save_path)
        
        logger.info(f"Random Forest Model trained and saved at: {save_path}")