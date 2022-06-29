import pandas as pd
from ner.config.configurations import Configuration
from ner.exception.exception import CustomException
from ner.entity.config_entity import DataValidationConfig
from ner.components.data_ingestion import DataIngestion
from typing import Dict, List
import logging
import sys

logger = logging.getLogger(__name__)


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data: Dict):
        logger.info(" Data Validation Started ")
        self.data_validation_config = data_validation_config
        self.data = data

    def check_columns_names(self) -> bool:
        try:
            logger.info(" Checking Columns of all the splits ")
            column_check_result = list()

            for split_name in self.data_validation_config.data_split:
                column_check_result.append(
                    sum(pd.DataFrame(self.data[split_name]).columns == self.data_validation_config.columns_check)
                )

            logger.info(f" Check Results {column_check_result}")

            if sum(column_check_result) == len(self.data_validation_config.data_split) * \
                    len(self.data_validation_config.columns_check):
                return True
            else:
                return False

        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def type_check(self) -> bool:
        try:
            """ Implement Type Check Here """
            logger.info(" Checking type check of all the splits ")
            result = self.data_validation_config.type_check
            return True
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def null_check(self) -> bool:
        try:
            """ Implement null Check Here """
            logger.info(" Checking null check of all the splits ")
            result = self.data_validation_config.null_check
            return True
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def drive_checks(self) -> List[List[bool]]:
        logger.info(" Checks Initiated  ")
        checks = list()
        checks.append(
            [
                self.check_columns_names(),
                self.type_check(),
                self.null_check()
            ]
        )
        logger.info(f" Checks Completed Result : {checks}")
        return checks


if __name__ == "__main__":
    project_config = Configuration()
    ingestion = DataIngestion(project_config.get_data_ingestion_config())
    en_data = ingestion.get_data()

    validate = DataValidation(data_validation_config=project_config.get_data_validation_config()
                              , data=en_data)
    check = validate.drive_checks()
    print(check)
