#pipeline
from src.ML_LAB.constants import *
from src.ML_LAB.utils.common import read_yaml, create_directories
from src.ML_LAB.entity.config_entity import *
from src.ML_LAB.config.configuration import *
from src.ML_LAB.components.Model_evaluation import *

STAGE_NAME = "Model evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.save_results()