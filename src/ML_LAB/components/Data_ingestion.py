
import os
import numpy as np
import pandas as pd
from src.ML_LAB.utils.common import get_size
from src.ML_LAB.entity.config_entity import *
from src.ML_LAB import logger


class Dataingestion:
    def __init__(self, config : DataIngestionConfig):
        self.config = config

    def data_file(self):
        local_path = self.config.local_data_file

        os.makedirs(os.path.dirname(local_path), exist_ok = True)

        print("Data infestion file made")