import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self, df):
        try:
            null_columns = df.columns[df.isnull().any()]
            skew_values = df[null_columns].skew(numeric_only=True)
            symmetric_columns = skew_values[skew_values.between(-0.5, 0.5)].index.tolist()
            skewed_columns = [col for col in null_columns if col not in symmetric_columns]

            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            untouched_columns = [col for col in numerical_columns if col not in null_columns]

            mean_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='mean')),
                ("scaler", StandardScaler())
            ])

            median_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            scale_only_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("mean_pipe", mean_pipeline, symmetric_columns),
                ("median_pipe", median_pipeline, skewed_columns),
                ("scale_only", scale_only_pipeline, untouched_columns),
            ])

            return preprocessor
        except Exception as e:
            raise e     
  
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column_name = "equipment_energy_consumption"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Get preprocessor using training data
            preprocessing_obj = self.get_data_transformation_object(input_feature_train_df)

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise e
