# components

import urllib.request as request
import pandas as pd
import os
from src.ML_LAB import logger
from src.ML_LAB.utils.common import get_size
from src.ML_LAB.entity.config_entity import *

class DataValidation:
    def __init__(self, config : DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = None
            data = pd.read_csv(r"D:\Cdac_ML\Assignments\Lab_assesment\artifacts\data_ingestion\hotel.csv")
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e
        