import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from source.exception import CustomException
from source.logger import logging
from source.components.data_transformation import DataTransformationConfig
from source.components.data_transformation import DataTransformation

from source.components.model_training import ModelTrainerConfig
from source.components.model_training import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entering the Data Ingestion method or Component')
        try:
            # File path for the raw dataset
            file_path = 'notebook/data/Fuel Cell.csv'
            logging.info(f'Reading data from: {file_path}')
            df = pd.read_csv(file_path)
            logging.info('Successfully read the data into a DataFrame')

            # Create directories if they don't exist
            data_dir = os.path.dirname(self.ingestion_config.train_data_path)
            logging.info(f"Creating directory: {data_dir}")
            os.makedirs(data_dir, exist_ok=True)

            # Save raw artifacts
            logging.info(f"Saving raw data to: {self.ingestion_config.raw_data_path}")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Train-test split
            logging.info('Initiating Train-Test Split')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test sets
            logging.info(f"Saving train artifacts to: {self.ingestion_config.train_data_path}")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            logging.info(f"Saving test artifacts to: {self.ingestion_config.test_data_path}")
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion process completed successfully')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error('Error occurred during the Data Ingestion process')
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logging.info("Starting the Data Ingestion process")
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion Completed Successfully.\nTrain Data Path: {train_data}\nTest Data Path: {test_data}")

        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data, test_data)


        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr)) 


        
    except Exception as e:
        logging.exception("An error occurred during the Data Ingestion process.")
