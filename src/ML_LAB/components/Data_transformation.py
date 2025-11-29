import os
import pandas as pd
import numpy as np
from src.ML_LAB import logger
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

class DataTransformationConfig:
    def __init__(self, root_dir, data_path):
        self.root_dir = root_dir
        self.data_path = data_path

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def drop_high_frequency_columns(self, df, threshold=0.95):
        """
        Drop columns where one value dominates > threshold (e.g. 95%)
        """
        cols_to_drop = []
        for col in df.columns:
            if df[col].dtype == 'object': # Only check categorical
                freq = (df[col].value_counts(normalize=True).max())
                if freq >= threshold:
                    cols_to_drop.append(col)
                    logger.info(f"Dropping column '{col}' due to high frequency ({freq:.2f})")
        
        return cols_to_drop

    def drop_high_vif_features(self, X_train_num, vif_threshold=20):
        """
        Iteratively drops features with high VIF.
        """
        cols_to_drop = []
        X_temp = X_train_num.copy()
        
        X_temp['intercept'] = 1 

        while True:
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_temp.columns
            vif_data["VIF"] = [variance_inflation_factor(X_temp.values, i) 
                               for i in range(X_temp.shape[1])]
            
            vif_data = vif_data[vif_data['feature'] != 'intercept']
            
            max_vif = vif_data['VIF'].max()
            if max_vif > vif_threshold:
                feature_to_drop = vif_data.sort_values('VIF', ascending=False)['feature'].iloc[0]
                cols_to_drop.append(feature_to_drop)
                X_temp.drop(columns=[feature_to_drop], inplace=True)
                logger.info(f"Dropping column '{feature_to_drop}' due to VIF: {max_vif:.2f}")
            else:
                break
                
        return cols_to_drop

    def initiate_data_transformation(self):
        try:
            logger.info("Loading data...")
            df = pd.read_csv(self.config.data_path)

            if 'Booking_ID' in df.columns:
                df.drop('Booking_ID', axis=1, inplace=True)

            logger.info("Splitting data into Train and Test...")
            target_col = "booking_status"
            
            X = df.drop(target_col, axis=1)
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            encoding_map = {'Canceled': 1, 'Not_Canceled': 0}
            y_train = y_train.map(encoding_map)
            y_test = y_test.map(encoding_map)

            cat_cols_train = X_train.select_dtypes(include=['object']).columns
            freq_drop_cols = self.drop_high_frequency_columns(X_train[cat_cols_train])
            
            X_train.drop(columns=freq_drop_cols, inplace=True)
            X_test.drop(columns=freq_drop_cols, inplace=True) # Apply same drop to test

            num_cols_train = X_train.select_dtypes(include=['number']).columns
            vif_drop_cols = self.drop_high_vif_features(X_train[num_cols_train])
            
            X_train.drop(columns=vif_drop_cols, inplace=True)
            X_test.drop(columns=vif_drop_cols, inplace=True) # Apply same drop to test

            num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
            cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            logger.info(f"Categorical cols: {cat_cols}")
            logger.info(f"Numerical cols: {num_cols}")

            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            ohe.fit(X_train[cat_cols])

            X_train_cat = pd.DataFrame(ohe.transform(X_train[cat_cols]), columns=ohe.get_feature_names_out(cat_cols), index=X_train.index)
            X_test_cat = pd.DataFrame(ohe.transform(X_test[cat_cols]), columns=ohe.get_feature_names_out(cat_cols), index=X_test.index)

            scaler = StandardScaler()
            scaler.fit(X_train[num_cols])

            X_train_num = pd.DataFrame(scaler.transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
            X_test_num = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

       
            train_final = pd.concat([X_train_num, X_train_cat, y_train], axis=1)
            test_final = pd.concat([X_test_num, X_test_cat, y_test], axis=1)

      
            train_file_path = os.path.join(self.config.root_dir, "train.csv")
            test_file_path = os.path.join(self.config.root_dir, "test.csv")

            train_final.to_csv(train_file_path, index=False)
            test_final.to_csv(test_file_path, index=False)

            logger.info(f"Transformation Complete. Train Shape: {train_final.shape}, Test Shape: {test_final.shape}")
            print(f"Train Saved at: {train_file_path}")
            print(f"Test Saved at: {test_file_path}")

            preprocessor_obj = {
                "scaler": scaler,
                "ohe": ohe,
                "num_cols": num_cols,
                "cat_cols": cat_cols,
                "freq_drop_cols": freq_drop_cols,
                "vif_drop_cols": vif_drop_cols
            }
            
            # Save preprocessor.obj
            preprocessor_path = os.path.join(self.config.root_dir, "preprocessor.obj")
            joblib.dump(preprocessor_obj, preprocessor_path)
            
            logger.info(f"Preprocessor object saved at {preprocessor_path}")
            # ------------------------------------

            logger.info(f"Transformation Complete. Train Shape: {train_final.shape}, Test Shape: {test_final.shape}")

        except Exception as e:
            logger.error(f"Error in Data Transformation: {str(e)}")
            raise e