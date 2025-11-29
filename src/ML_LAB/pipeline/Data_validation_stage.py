#pipeline
from src.ML_LAB.constants import *
from src.ML_LAB.utils.common import read_yaml, create_directories
from src.ML_LAB.entity.config_entity import *
from src.ML_LAB.config.configuration import *
from src.ML_LAB.components.Data_validation import *

STAGE_NAME = "Data Validation stage"

class DataValidationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()