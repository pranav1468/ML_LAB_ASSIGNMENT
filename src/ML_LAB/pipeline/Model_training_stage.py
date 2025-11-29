#pipeline
from src.ML_LAB.constants import *
from src.ML_LAB.utils.common import read_yaml, create_directories
from src.ML_LAB.entity.config_entity import *
from src.ML_LAB.config.configuration import *
from src.ML_LAB.components.Model_training import *

STAGE_NAME = "Model Trainer Stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_tr_config = ModelTrainer(config=model_trainer_config)
        model_tr_config.train()