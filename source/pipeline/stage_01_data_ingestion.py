from source.components.data_ingestion import DataIngestion
from source import logger
from source import exception

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass


if __name__ == '__main__':
    try:
        
        obj = DataIngestionTrainingPipeline()
        obj.main()

    except Exception as e:
        logger.exception(e)
        raise e