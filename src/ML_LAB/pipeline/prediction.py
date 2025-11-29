import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os

class PredictionPipeline:
    def __init__(self):
        # Paths check kar lena, agar change ho to update kar dena
        self.model_path = Path("artifacts/model_trainer/model.joblib") 
        self.preprocessor_path = Path("artifacts/data_transformation/preprocessor.obj")

    def predict(self, features):
        try:
            # 1. Load Model & Preprocessor
            model = joblib.load(self.model_path)
            preprocessor = joblib.load(self.preprocessor_path)
            
            scaler = preprocessor['scaler']
            ohe = preprocessor['ohe']
            num_cols = preprocessor['num_cols']
            cat_cols = preprocessor['cat_cols']

            # 2. Create DataFrame
            df = pd.DataFrame(features)
            
            # 3. Scaling (Numerical)
            df_num = pd.DataFrame(scaler.transform(df[num_cols]), columns=num_cols)
            
            # 4. Encoding (Categorical)
            df_cat = pd.DataFrame(ohe.transform(df[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
            
            # 5. Concatenate
            df_final = pd.concat([df_num, df_cat], axis=1)
            
            # 6. Predict
            prediction = model.predict(df_final)
            return prediction

        except Exception as e:
            raise e