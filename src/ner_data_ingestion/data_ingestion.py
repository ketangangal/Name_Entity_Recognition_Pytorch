from datasets import load_dataset
from datasets import get_dataset_config_names
from src.app_logger.logger import CustomLogger
from src.app_exception_handler.exception import CustomException
import sys

logger = CustomLogger("logs")


class DataIngestion:
    def __init__(self):
        self.dataset_name = "xtreme"

    def get_data(self):
        try:
            """
            Meta-Info: Cross-lingual TRansfer Evaluation of Multilingual Encoders 
                       (XTREME) benchmark called WikiANN or PAN-X.
            """
            ner_data_subsets = get_dataset_config_names(self.dataset_name)
            logger.info(f"XTREME has {len(ner_data_subsets)} configurations")

            subsets = [s for s in ner_data_subsets if s.startswith("PAN")]
            logger.info(f"XTREME has {subsets} Subsets Taking English For Fine Tuning")

            en_data = load_dataset("xtreme", name="PAN-X.en")
            logger.info(f"Dataset Info : {en_data}")

            return en_data

        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)




