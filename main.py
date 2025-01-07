from source.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from source import logger
import ssl

STAGE_NAME = "Data Ingestion stage"
ssl_context = ssl._create_unverified_context()

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise