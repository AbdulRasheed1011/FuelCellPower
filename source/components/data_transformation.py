import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import datetime
import calendar

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def preprocess_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles datetime column transformation and feature extraction.
        """
        try:
            logging.info("Converting Datetime column to pandas datetime format")
            df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d-%m-%Y %H:%M", errors="coerce")

            # Extract features from Datetime
            logging.info("Extracting Date, Month, Year, and Time from Datetime")
            df["Date"] = df["Datetime"].dt.day.fillna(31).astype(int)
            df["Month"] = df["Datetime"].dt.month.fillna(12).astype(int)
            df["Year"] = df["Datetime"].dt.year.fillna(2017).astype(int)
            df["Time"] = df["Datetime"].dt.time

            # Map month numbers to names
            df["Month"] = df["Month"].map(lambda x: calendar.month_name[x] if pd.notna(x) else None)

            # Assign target variable
            df["Power"] = df['Fuel Cell Power '].fillna(1614).astype(int)

            # Drop unnecessary columns
            logging.info("Dropping original Datetime and Fuel Cell Power columns")
            df.drop(columns=["Datetime", "Fuel Cell Power "], axis=1, inplace=True)

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self, num_columns, cat_columns):
        """
        Creates a ColumnTransformer object for preprocessing.
        """
        try:
            logging.info("Creating preprocessing pipelines for numerical and categorical features")

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_columns),
                    ('cat_pipeline', cat_pipeline, cat_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Preprocessing datetime features in train and test datasets")
            train_df = self.preprocess_datetime_features(train_df)
            test_df = self.preprocess_datetime_features(test_df)

            target_column = 'Power'
            num_columns = train_df.select_dtypes(exclude="object").drop(columns=[target_column]).columns.tolist()
            cat_columns = train_df.select_dtypes(include="object").columns.tolist()

            logging.info("Obtaining preprocessor object")
            preprocessing_obj = self.get_data_transformer_object(num_columns, cat_columns)

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(f"Shape of input_feature_train_df: {input_feature_train_df.shape}")
            logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")

            # Preprocess features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            logging.info(f"Shape of target_feature_train_df (after conversion): {np.array(target_feature_train_df).shape}")

            # Ensure dimensions match
            if input_feature_train_arr.shape[0] != target_feature_train_df.shape[0]:
                raise ValueError("Mismatch in rows between input features and target for training data.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error("An error occurred during the Data Transformation process.")
            raise CustomException(e, sys)
