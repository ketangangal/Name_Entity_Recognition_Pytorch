from ner.config.configurations import Configuration
from ner.components.data_ingestion import DataIngestion
from ner.components.data_validation import DataValidation
from ner.components.data_prepration import DataPreprocessing
from ner.components.model_training import TrainTokenClassifier
from ner.exception.exception import CustomException
from typing import Any, Dict, List, ClassVar
import logging
import sys

logger = logging.getLogger(__name__)


class TrainPipeline:
    def __init__(self, config):
        self.config = config

    def run_data_ingestion(self) -> Dict:
        try:
            logger.info(" Running Data Ingestion pipeline ")
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            data = data_ingestion.get_data()
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def run_data_validation(self, data) -> List[List[bool]]:
        try:
            logger.info(" Running Data validation Pipeline ")
            validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                        data=data)
            checks = validation.drive_checks()
            return checks
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def run_data_preparation(self, data) -> Dict:
        try:
            logger.info(" Running Data Preparation pipeline ")
            data_preprocessng = DataPreprocessing(data_preprocessing_config=self.config.get_data_preprocessing_config(),
                                                  data=data)
            data = data_preprocessng.prepare_data_for_fine_tuning()
            return data
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def run_model_training(self, data):
        try:
            logger.info(" Run model Training ")
            classifier = TrainTokenClassifier(model_training_config=self.config.get_model_train_pipeline_config(),
                                              processed_data=data)
            classifier.train()

            logger.info(" Training Completed ")

        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def run_pipeline(self):
        data = self.run_data_ingestion()
        checks = self.run_data_validation(data=data)
        if sum(checks[0]) == 3:
            logger.info("Checks Completed")
            processed_data = self.run_data_preparation(data=data)
            logger.info(f"Preprocessed Data {processed_data}")
            self.run_model_training(data=processed_data)
        else:
            logger.error("Checks Failed")


if __name__ == "__main__":
    pipeline = TrainPipeline(Configuration())
    pipeline.run_pipeline()


