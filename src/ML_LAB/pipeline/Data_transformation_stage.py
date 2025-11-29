#pipeline
from src.ML_LAB.constants import *
from src.ML_LAB.utils.common import read_yaml, create_directories
from src.ML_LAB.entity.config_entity import *
from src.ML_LAB.config.configuration import *
from src.ML_LAB.components.Data_transformation import *

STAGE_NAME = "Data Transformation stage"

class DataTransformationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.initiate_data_transformation()