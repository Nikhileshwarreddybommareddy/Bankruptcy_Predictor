import sys,os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from imblearn.over_sampling import SMOTE

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    project_root: Path = Path(__file__).resolve().parent.parent.parent  # Adjust path as needed
    train_data_path: Path = project_root / "data/processed/train_transformed.npy"
    test_data_path: Path = project_root / "data/processed/test_transformed.npy"

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
            from src.components.data_ingestion import DataIngestionConfig
            """
            Here we first recognised numerical and categorical values
            created 2 pipelines for each
            num_pipeline will take numerical columns replace nulls with median and perform standard scaling
            cat_pipelines will take in categorical columns and perform imputing with mode, perform one_hot encoding and then scale
            In the end we wrap both of them in a column transformer for accessing them again and again.
            """
            try:
                train_df = pd.read_csv(DataIngestionConfig.train_data_path).drop(columns="Bankrupt",axis=1)
                numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_columns = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
                # here we are imputing the nulls with median values and we are applying standard scaler ofr numerical columns
                num_pipeline= Pipeline(
                    steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())

                    ]
                )
                # here most frequent means mode we  are replacing nulls with mode for categorical pipeline
                cat_pipeline=Pipeline(
                    steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                    ]
                )

                logging.info(f"Categorical columns: {categorical_columns}")
                logging.info(f"Numerical columns: {numerical_columns}")

                preprocessor=ColumnTransformer(
                    [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                    ]
                )

                return preprocessor
        
            except Exception as e:
                raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
         try:
              logging.info("Reading train and test data completed")
              train_df = pd.read_csv(train_path)
              test_df = pd.read_csv(test_path)
              logging.info("Reading train and test data completed")
              preprocessing_obj = self.get_data_transformer_object()
              logging.info("Initiated preprocessor")
              target_column_name="Bankrupt"
              input_feature_train_df = train_df.drop(columns=target_column_name,axis=1)
              target_feature_train_df = train_df[target_column_name]
              smote = SMOTE(sampling_strategy='auto', random_state=42)
              input_feature_test_df = test_df.drop(columns=target_column_name,axis=1)
              target_feature_test_df = test_df[target_column_name]
              logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
              input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
              input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
              input_feature_train_arr, target_feature_train_df = smote.fit_resample(input_feature_train_arr, target_feature_train_df)
              input_feature_test_arr, target_feature_test_df = smote.fit_resample(input_feature_test_arr, target_feature_test_df)
              train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
                ]
              test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
              np.save(self.data_transformation_config.train_data_path, train_arr)
              np.save(self.data_transformation_config.test_data_path, test_arr)
              logging.info(f"Saved preprocessing object.")
              save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)

         except Exception as e:
                raise CustomException(e,sys)