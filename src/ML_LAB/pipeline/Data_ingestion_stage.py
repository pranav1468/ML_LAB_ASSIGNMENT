#pipeline
from src.ML_LAB.constants import *
from src.ML_LAB.utils.common import read_yaml, create_directories
from src.ML_LAB.entity.config_entity import *
from src.ML_LAB.config.configuration import *
from src.ML_LAB.components.Data_ingestion import *

STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = Dataingestion(config = data_ingestion_config)
        data_ingestion.data_file()