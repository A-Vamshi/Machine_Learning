import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path : str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformationConfig = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
            This function is responsible for data transformation
        '''
        try:
            numerical_features = ["attribute1", "attribute2"] # write the attribute names / column names for numerical features
            categorical_features = ["attribute1", "attribute2"] # write the attribute names / column names for categorical features (non numerical) [may include things like gender or whatever]
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical and categorical encoding, handling missing values and scaling completed")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_features),
                    ("categorical_pipeline", categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test dfs are read")

            logging.info("Obtaining preprossesor object")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "target_columm_name" # write the target column name / attribute name 
            numerical_features = ["attribute1", "attribute2"] # write the attribute names / column names for numerical features
            categorical_features = ["attribute1", "attribute2"] # write the attribute names / column names for categorical features (non numerical) [may include things like gender or whatever]
            
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Applied preprocessing to training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_features_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.transformationConfig.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformationConfig.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys) 
        
